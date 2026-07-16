//! Industrial termbase: VI/EN/ZH/KO parallel glossary (safety commands,
//! machine names, logistics terms) — the "Industrial glossary engine"
//! differentiator from CLAUDE.md. This is the lookup table, loaded for
//! real from `data/glossary/termbase.tsv`.
//!
//! **Wired into `mt`'s decode** via `find_matches` (below) — see
//! `src/stages/mt.rs`'s module docs for the decoder-logit-bias
//! technique that consumes it. Placeholder substitution (replace the
//! source term with a placeholder string, translate, swap the correct
//! target term back in) was tried first and rejected: tested against
//! the real vi-en Opus-MT model with several placeholder candidates,
//! survival was inconsistent (`QQQZZZ` and `Zzz9` came through
//! verbatim, `XNAAA`, `FORKLIFTTOKEN`, and `xxxxx` all got silently
//! mangled or dropped, depending on how the model's subword tokenizer
//! happens to split each string). See
//! docs/decisions/glossary-runtime-enforcement-2026-07.md for both that
//! rejected approach and the decoder-bias technique that replaced it.

use std::collections::HashMap;
use std::path::Path;

use crate::error::OneVoiceError;

type TermKey = (String, String, String); // (source_lang, target_lang, source_term.lowercase)

#[derive(Debug, Default)]
pub struct Glossary {
    terms: HashMap<TermKey, String>,
}

impl Glossary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads `data_dir/termbase.tsv`: a header row `concept\t<lang>\t<lang>...`
    /// followed by one row per concept, e.g. `forklift\txe nâng\tforklift\t叉车\t지게차`.
    /// Inserts every directional pair among the languages present (VI->EN,
    /// EN->VI, VI->ZH, ... — whatever columns the file has, no code
    /// change needed to add a language). Missing file is not an error —
    /// returns an empty glossary, since the pipeline should still run
    /// (just without glossary data) rather than fail to start.
    pub fn load(data_dir: &Path) -> Result<Self, OneVoiceError> {
        let path = data_dir.join("termbase.tsv");
        if !path.exists() {
            return Ok(Self::new());
        }

        let contents = std::fs::read_to_string(&path)
            .map_err(|source| OneVoiceError::Glossary(format!("failed to read {}: {source}", path.display())))?;

        let mut lines = contents.lines();
        let header = lines
            .next()
            .ok_or_else(|| OneVoiceError::Glossary(format!("{} is empty", path.display())))?;
        let langs: Vec<&str> = header.split('\t').skip(1).collect(); // skip the "concept" column

        let mut glossary = Self::new();
        for (i, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row_num = i + 2; // 1-indexed, plus the header row
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != langs.len() + 1 {
                return Err(OneVoiceError::Glossary(format!(
                    "{}:{row_num}: expected {} columns, got {}",
                    path.display(),
                    langs.len() + 1,
                    fields.len()
                )));
            }
            let terms = &fields[1..];

            for (i, &source_lang) in langs.iter().enumerate() {
                for (j, &target_lang) in langs.iter().enumerate() {
                    if i != j {
                        glossary.insert(source_lang, target_lang, terms[i], terms[j]);
                    }
                }
            }
        }

        Ok(glossary)
    }

    pub fn insert(&mut self, source_lang: &str, target_lang: &str, source_term: &str, target_term: &str) {
        self.terms.insert(
            key(source_lang, target_lang, source_term),
            target_term.to_string(),
        );
    }

    pub fn lookup(&self, source_lang: &str, target_lang: &str, source_term: &str) -> Option<&str> {
        self.terms
            .get(&key(source_lang, target_lang, source_term))
            .map(String::as_str)
    }

    /// Finds every glossary entry for `(source_lang, target_lang)` whose
    /// source term occurs (case-insensitively) as a substring of `text`.
    /// Used by `mt`'s decode to decide which target terms to bias
    /// toward — see `src/stages/mt.rs`.
    pub fn find_matches(&self, source_lang: &str, target_lang: &str, text: &str) -> Vec<(&str, &str)> {
        let text_lower = text.to_lowercase();
        self.terms
            .iter()
            .filter(|((sl, tl, source_term), _)| sl == source_lang && tl == target_lang && text_lower.contains(source_term.as_str()))
            .map(|((_, _, source_term), target_term)| (source_term.as_str(), target_term.as_str()))
            .collect()
    }
}

fn key(source_lang: &str, target_lang: &str, source_term: &str) -> TermKey {
    (
        source_lang.to_string(),
        target_lang.to_string(),
        source_term.to_lowercase(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_case_insensitive() {
        let mut g = Glossary::new();
        g.insert("en", "vi", "Forklift", "xe nâng");
        assert_eq!(g.lookup("en", "vi", "forklift"), Some("xe nâng"));
    }

    #[test]
    fn lookup_is_scoped_to_language_pair() {
        let mut g = Glossary::new();
        g.insert("en", "vi", "forklift", "xe nâng");
        assert_eq!(g.lookup("en", "zh", "forklift"), None);
    }

    #[test]
    fn load_returns_empty_glossary_when_file_is_missing() {
        let dir = std::env::temp_dir().join(format!("onevoice-glossary-test-missing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let g = Glossary::load(&dir).expect("load should not error on a missing file");
        assert_eq!(g.lookup("vi", "en", "xe nâng"), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_parses_a_real_termbase_file_in_all_directions() {
        let dir = std::env::temp_dir().join(format!("onevoice-glossary-test-real-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("termbase.tsv"),
            "concept\tvi\ten\tzh\n\
             forklift\txe nâng\tforklift\t叉车\n",
        )
        .unwrap();

        let g = Glossary::load(&dir).expect("load failed");
        assert_eq!(g.lookup("vi", "en", "xe nâng"), Some("forklift"));
        assert_eq!(g.lookup("en", "vi", "forklift"), Some("xe nâng"));
        assert_eq!(g.lookup("en", "zh", "forklift"), Some("叉车"));
        assert_eq!(g.lookup("vi", "zh", "xe nâng"), Some("叉车"));
        assert_eq!(g.lookup("vi", "ko", "xe nâng"), None); // ko not in this fixture's header

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn find_matches_returns_terms_present_in_the_text() {
        let mut g = Glossary::new();
        g.insert("vi", "en", "xe nâng", "forklift");
        g.insert("vi", "en", "báo động", "alarm");

        let matches = g.find_matches("vi", "en", "Xe nâng đang đến gần khu vực số ba.");
        assert_eq!(matches, vec![("xe nâng", "forklift")]);

        let no_matches = g.find_matches("vi", "en", "Xin chào, bạn khỏe không?");
        assert!(no_matches.is_empty());
    }

    #[test]
    fn load_rejects_a_malformed_row() {
        let dir = std::env::temp_dir().join(format!("onevoice-glossary-test-malformed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("termbase.tsv"), "concept\tvi\ten\nforklift\txe nâng\n").unwrap(); // missing the "en" column
        let err = Glossary::load(&dir).unwrap_err();
        assert!(matches!(err, OneVoiceError::Glossary(_)));
        std::fs::remove_dir_all(&dir).ok();
    }
}
