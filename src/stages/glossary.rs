//! Industrial termbase: VI/EN/ZH/KO parallel glossary (safety commands,
//! machine names, logistics terms) — the "Industrial glossary engine"
//! differentiator from CLAUDE.md. This is the lookup table; enforcement
//! during MT decode (constrained decoding) lives in `mt`, once that
//! stage is implemented against a real model.
//!
//! Term data lives in data/glossary/, loaded via `Glossary::load`.

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

    /// Loads a termbase from data/glossary/. Format/parsing TBD once the
    /// termbase schema (docs/glossary/) is settled — currently a no-op
    /// that returns an empty glossary so callers can wire against the
    /// real interface early.
    pub fn load(_data_dir: &Path) -> Result<Self, OneVoiceError> {
        Ok(Self::new())
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
}
