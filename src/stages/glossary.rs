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
use std::fmt;
use std::path::Path;

use crate::error::OneVoiceError;

type TermKey = (String, String, String); // (source_lang, target_lang, source_term.lowercase)

#[derive(Default)]
pub struct Glossary {
    terms: HashMap<TermKey, String>,
    searchers: HashMap<(String, String), (aho_corasick::AhoCorasick, Vec<String>)>,
}

impl fmt::Debug for Glossary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Glossary")
            .field("terms", &self.terms)
            .finish_non_exhaustive()
    }
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

        let contents = std::fs::read_to_string(&path).map_err(|source| {
            OneVoiceError::Glossary(format!("failed to read {}: {source}", path.display()))
        })?;

        let mut lines = contents.lines();
        let header = lines.next().ok_or_else(|| {
            OneVoiceError::Glossary("termbase.tsv is empty (no header row)".to_string())
        })?;

        let langs: Vec<&str> = header.split('\t').skip(1).collect();
        if langs.is_empty() {
            return Err(OneVoiceError::Glossary(
                "no language columns found in termbase header".to_string(),
            ));
        }

        let mut glossary = Self::new();

        for (row_idx, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != langs.len() + 1 {
                return Err(OneVoiceError::Glossary(format!(
                    "row {} has {} columns, expected {} (concept + {})",
                    row_idx + 1,
                    parts.len(),
                    langs.len() + 1,
                    langs.join(", ")
                )));
            }

            let terms = &parts[1..];

            for (i, &source_lang) in langs.iter().enumerate() {
                for (j, &target_lang) in langs.iter().enumerate() {
                    if i != j {
                        // Insert directly into the map to avoid rebuilding searchers on every inner iteration
                        glossary.terms.insert(
                            key(source_lang, target_lang, terms[i]),
                            terms[j].to_string(),
                        );
                    }
                }
            }
        }

        glossary.build_searchers();
        Ok(glossary)
    }

    #[allow(dead_code)]
    pub fn insert(
        &mut self,
        source_lang: &str,
        target_lang: &str,
        source_term: &str,
        target_term: &str,
    ) {
        self.terms.insert(
            key(source_lang, target_lang, source_term),
            target_term.to_string(),
        );
        self.build_searchers();
    }

    #[allow(dead_code)]
    pub fn lookup(&self, source_lang: &str, target_lang: &str, source_term: &str) -> Option<&str> {
        self.terms
            .get(&key(source_lang, target_lang, source_term))
            .map(String::as_str)
    }

    /// Finds every glossary entry for `(source_lang, target_lang)` whose
    /// source term occurs (case-insensitively) as a substring of `text`.
    /// Used by `mt`'s decode to decide which target terms to bias
    /// toward — see `src/stages/mt.rs`.
    pub fn find_matches(
        &self,
        source_lang: &str,
        target_lang: &str,
        text: &str,
    ) -> Vec<(&str, &str)> {
        let text_lower = text.to_lowercase();
        let pair_key = (source_lang.to_string(), target_lang.to_string());

        if let Some((ac, source_terms)) = self.searchers.get(&pair_key) {
            let mut matches = Vec::new();
            for mat in ac.find_iter(&text_lower) {
                let matched_source = &source_terms[mat.pattern()];
                let term_key = (
                    source_lang.to_string(),
                    target_lang.to_string(),
                    matched_source.clone(),
                );
                if let Some(target_term) = self.terms.get(&term_key) {
                    matches.push((matched_source.as_str(), target_term.as_str()));
                }
            }
            matches.sort_unstable();
            matches.dedup();
            matches
        } else {
            Vec::new()
        }
    }

    /// Compiles Aho-Corasick searchers for all populated translation directions.
    fn build_searchers(&mut self) {
        let mut groups: HashMap<(String, String), Vec<String>> = HashMap::new();
        for (sl, tl, source_term) in self.terms.keys() {
            groups
                .entry((sl.clone(), tl.clone()))
                .or_default()
                .push(source_term.clone());
        }

        self.searchers.clear();
        for (pair, mut source_terms) in groups {
            source_terms.sort_unstable_by_key(|s| std::cmp::Reverse(s.len()));
            source_terms.dedup();

            if let Ok(ac) = aho_corasick::AhoCorasick::builder()
                .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                .build(&source_terms)
            {
                self.searchers.insert(pair, (ac, source_terms));
            }
        }
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
        let dir = std::env::temp_dir().join(format!(
            "onevoice-glossary-test-missing-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let g = Glossary::load(&dir).expect("load should not error on a missing file");
        assert_eq!(g.lookup("vi", "en", "xe nâng"), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_parses_a_real_termbase_file_in_all_directions() {
        let dir = std::env::temp_dir().join(format!(
            "onevoice-glossary-test-real-{}",
            std::process::id()
        ));
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
        let dir = std::env::temp_dir().join(format!(
            "onevoice-glossary-test-malformed-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("termbase.tsv"),
            "concept\tvi\ten\nforklift\txe nâng\n",
        )
        .unwrap(); // missing the "en" column
        let err = Glossary::load(&dir).unwrap_err();
        assert!(matches!(err, OneVoiceError::Glossary(_)));
        std::fs::remove_dir_all(&dir).ok();
    }
}
