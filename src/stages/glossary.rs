//! Industrial termbase: VI/EN/ZH/KO parallel glossary (safety commands,
//! machine names, logistics terms) — the "Industrial glossary engine"
//! differentiator from AGENTS.md. This is the lookup table, loaded for
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

/// A per-direction lookup: the Aho-Corasick automaton plus a `(source,
/// target)` pair parallel to its pattern indices, so `find_matches` can
/// read the target straight off the match instead of rebuilding a
/// `TermKey` (three `String`s) per hit on the real-time path.
type DirectionSearcher = (aho_corasick::AhoCorasick, Vec<(String, String)>);

#[derive(Default)]
pub struct Glossary {
    terms: HashMap<TermKey, String>,
    searchers: HashMap<(String, String), DirectionSearcher>,
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

        if let Some((ac, pairs)) = self.searchers.get(&pair_key) {
            let mut matches = Vec::new();
            for mat in ac.find_iter(&text_lower) {
                if !is_word_bounded(&text_lower, mat.start(), mat.end()) {
                    continue;
                }
                let (source, target) = &pairs[mat.pattern()];
                matches.push((source.as_str(), target.as_str()));
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
        // Dedup by source term within each direction; a source term maps to
        // exactly one target per direction.
        let mut groups: HashMap<(String, String), HashMap<String, String>> = HashMap::new();
        for ((sl, tl, source_term), target_term) in &self.terms {
            groups
                .entry((sl.clone(), tl.clone()))
                .or_default()
                .insert(source_term.clone(), target_term.clone());
        }

        self.searchers.clear();
        for (pair, map) in groups {
            let mut pairs: Vec<(String, String)> = map.into_iter().collect();
            // Deterministic order for the pattern-index -> (source, target)
            // mapping (LeftmostLongest picks the longest match regardless).
            pairs.sort_unstable_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));

            let ac = {
                let patterns: Vec<&str> = pairs.iter().map(|(s, _)| s.as_str()).collect();
                aho_corasick::AhoCorasick::builder()
                    .match_kind(aho_corasick::MatchKind::LeftmostLongest)
                    .build(&patterns)
            };
            if let Ok(ac) = ac {
                self.searchers.insert(pair, (ac, pairs));
            }
        }
    }
}

/// True unless the match `text[start..end]` is immediately flanked by an
/// ASCII alphanumeric character. Deliberately ASCII-only rather than
/// Unicode `is_alphanumeric`: Vietnamese diacritics and CJK are letters
/// but don't delimit words the way Latin script does, so treating them as
/// boundary characters would break terms like "叉车" inside "大型叉车".
/// ASCII is enough to stop a short English term ("lift") matching inside a
/// longer word ("forklift", "shoplifting") as the termbase grows.
fn is_word_bounded(text: &str, start: usize, end: usize) -> bool {
    let before = text[..start].chars().next_back();
    let after = text[end..].chars().next();
    !before.is_some_and(|c| c.is_ascii_alphanumeric())
        && !after.is_some_and(|c| c.is_ascii_alphanumeric())
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
    fn find_matches_ignores_matches_inside_a_longer_ascii_word() {
        let mut g = Glossary::new();
        g.insert("en", "vi", "lift", "nâng");

        // "lift" inside "forklift"/"shoplifting" must not match.
        assert!(g.find_matches("en", "vi", "forklift").is_empty());
        assert!(g.find_matches("en", "vi", "shoplifting").is_empty());
        // Standalone (any non-alphanumeric flank) still matches.
        assert_eq!(
            g.find_matches("en", "vi", "please lift now"),
            vec![("lift", "nâng")]
        );
        assert_eq!(
            g.find_matches("en", "vi", "lift-off"),
            vec![("lift", "nâng")]
        );
    }

    #[test]
    fn find_matches_keeps_cjk_substrings_that_ascii_words_would_reject() {
        // CJK has no word-delimiting spaces; a term embedded in a longer
        // string is still a real match (the flanking char is CJK, not
        // ASCII alphanumeric, so the word-boundary check lets it through).
        let mut g = Glossary::new();
        g.insert("zh", "vi", "叉车", "xe nâng");
        assert_eq!(
            g.find_matches("zh", "vi", "大型叉车正在靠近"),
            vec![("叉车", "xe nâng")]
        );
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
