from typing import List, Dict, Any
from config import (
    MIN_RECOMMENDED_WORDS, MAX_RECOMMENDED_WORDS,
    IDEAL_KEYWORD_DENSITY_MIN, IDEAL_KEYWORD_DENSITY_MAX,
    IDEAL_PASSIVE_VOICE_MAX, IDEAL_READABILITY_FLESCH_MIN,
    IDEAL_LEXICAL_DIVERSITY_MIN, MIN_HEADING_DENSITY
)

class RecommendationEngine:
    """
    Evaluates extracted NLP features and generates actionable improvements 
    for SEO, readability, style, and vocabulary structure.
    """
    
    def generate_recommendations(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        recs = []
        
        word_count = features.get("word_count", 0.0)
        sentence_count = features.get("sentence_count", 1.0)
        
        # 1. Content Length / Structure (SEO)
        if word_count < MIN_RECOMMENDED_WORDS:
            recs.append({
                "category": "SEO",
                "priority": "High",
                "action": "Increase content length",
                "explanation": f"Your text has only {int(word_count)} words. Standard SEO articles should aim for at least {MIN_RECOMMENDED_WORDS} words to provide enough depth.",
                "impact": "Longer, comprehensive content ranks higher on search engines and answers search intent better."
            })
        elif word_count > MAX_RECOMMENDED_WORDS:
            recs.append({
                "category": "SEO",
                "priority": "Low",
                "action": "Structure long-form content",
                "explanation": f"Your text has {int(word_count)} words. Very long articles can become fatiguing unless structured with a table of contents or split into a series.",
                "impact": "Improves readability, prevents bounce rates, and allows targeting multiple subtopics efficiently."
            })
            
        # Heading Density
        heading_count = features.get("heading_count", 0.0)
        ideal_headings = max(1.0, word_count / MIN_HEADING_DENSITY)
        if heading_count < ideal_headings and word_count >= 200:
            recs.append({
                "category": "SEO",
                "priority": "High",
                "action": "Add subheadings (H2/H3)",
                "explanation": f"You have {int(heading_count)} headings for a {int(word_count)}-word article. We recommend at least {int(ideal_headings)} subheadings to break up the text.",
                "impact": "Headings define content hierarchy, helping readers scan the article and search engines index the sections."
            })
            
        # 2. Readability & Complexity
        flesch_ease = features.get("readability_flesch_reading_ease", 100.0)
        fk_grade = features.get("readability_flesch_kincaid_grade", 0.0)
        
        if flesch_ease < IDEAL_READABILITY_FLESCH_MIN:
            recs.append({
                "category": "Readability",
                "priority": "High",
                "action": "Improve reading ease (Flesch Score)",
                "explanation": f"Flesch Reading Ease score is {flesch_ease:.1f} (target: > {IDEAL_READABILITY_FLESCH_MIN}). The grade level is {fk_grade:.1f}, which is difficult for general audiences.",
                "impact": "Clear, readable content decreases bounce rates, increases average session duration, and is preferred by search engines."
            })
            
        # Sentence length
        avg_sent_len = features.get("avg_sentence_length", 0.0)
        if avg_sent_len > 22.0:
            recs.append({
                "category": "Readability",
                "priority": "Medium",
                "action": "Shorten long sentences",
                "explanation": f"Average sentence length is {avg_sent_len:.1f} words. Sentences longer than 20 words degrade readability.",
                "impact": "Shorter sentences improve comprehension, particularly for mobile readers."
            })
            
        # 3. Writing Style & Grammar
        passive_pct = features.get("passive_voice_pct", 0.0)
        if passive_pct > IDEAL_PASSIVE_VOICE_MAX:
            recs.append({
                "category": "Writing Style",
                "priority": "Medium",
                "action": "Reduce passive voice usage",
                "explanation": f"Around {passive_pct:.1f}% of your sentences are in the passive voice. The standard target is below {IDEAL_PASSIVE_VOICE_MAX}%.",
                "impact": "Active voice is more engaging, clear, direct, and conveys authority."
            })
            
        # Spelling mistake ratio
        spelling_ratio = features.get("spelling_mistake_ratio", 0.0) * 100.0
        if spelling_ratio > 1.5:  # more than 1.5% spelling errors
            recs.append({
                "category": "Writing Style",
                "priority": "High",
                "action": "Proofread for spelling mistakes",
                "explanation": f"Estimated spelling mistake rate is {spelling_ratio:.1f}% ({int(features.get('spelling_mistakes', 0))} errors detected).",
                "impact": "Typographic and spelling errors severely damage user trust and content authority."
            })
            
        # 4. Vocabulary & Lexical Diversity
        lex_div = features.get("lexical_diversity", 1.0)
        if lex_div < IDEAL_LEXICAL_DIVERSITY_MIN and word_count >= 150:
            recs.append({
                "category": "Vocabulary",
                "priority": "Medium",
                "action": "Enhance vocabulary diversity",
                "explanation": f"Your lexical diversity (Type-Token Ratio) is {lex_div:.2f}. We recommend a score of at least {IDEAL_LEXICAL_DIVERSITY_MIN}.",
                "impact": "Rich vocabulary prevents repetitive phrasing and signals high-quality content to semantic search engines."
            })
            
        # 5. SEO Linking and Media
        total_links = features.get("total_links", 0.0)
        if total_links == 0 and word_count >= 200:
            recs.append({
                "category": "SEO",
                "priority": "Medium",
                "action": "Add internal and external links",
                "explanation": "No links were detected in the text. Linking to related resources adds credibility.",
                "impact": "Links pass domain authority, index webpages faster, and guide users to relevant citations."
            })
            
        # Images
        image_count = features.get("image_count", 0.0)
        if image_count == 0 and word_count >= 400:
            recs.append({
                "category": "SEO",
                "priority": "Low",
                "action": "Integrate visual assets",
                "explanation": "No image references were detected. Visually engaging posts are highly appreciated.",
                "impact": "Images split wall-of-text layouts, improve session times, and rank in image search engines."
            })
            
        # 6. Keyword Optimization
        keyword_density = features.get("keyword_density", 0.0)
        if keyword_density < IDEAL_KEYWORD_DENSITY_MIN and word_count >= 200:
            recs.append({
                "category": "SEO",
                "priority": "Medium",
                "action": "Optimize keyword density",
                "explanation": f"Keyword density is {keyword_density:.1f}% which is below the recommended {IDEAL_KEYWORD_DENSITY_MIN}%. Make sure to introduce the target topic.",
                "impact": "Helps search engines understand the exact search terms your article matches."
            })
        elif keyword_density > IDEAL_KEYWORD_DENSITY_MAX:
            recs.append({
                "category": "SEO",
                "priority": "High",
                "action": "Reduce keyword stuffing",
                "explanation": f"Your keyword density is {keyword_density:.1f}%. Exceeding {IDEAL_KEYWORD_DENSITY_MAX}% triggers spam flags.",
                "impact": "Avoiding over-optimization protects your article from ranking penalties and ensures standard reading flow."
            })
            
        # Transition words
        trans_density = features.get("transition_word_density", 0.0)
        if trans_density < 2.0:
            recs.append({
                "category": "Readability",
                "priority": "Low",
                "action": "Utilize transition words",
                "explanation": f"Transition words (connectors) represent only {trans_density:.1f}% of the text.",
                "impact": "Transition words connect ideas logically, improving flow and readability."
            })
            
        # 7. Tone and Sentiment
        subjectivity = features.get("sentiment_subjectivity", 0.5)
        if subjectivity > 0.7:
            recs.append({
                "category": "Tone",
                "priority": "Low",
                "action": "Establish objective authority",
                "explanation": f"The subjectivity score is {subjectivity:.2f}, indicating highly opinionated language. Balanced analysis is preferred for informational articles.",
                "impact": "Objective writing increases academic trust and brand credibility."
            })
            
        return recs

if __name__ == "__main__":
    engine = RecommendationEngine()
    dummy_feats = {
        "word_count": 150.0,
        "heading_count": 0.0,
        "readability_flesch_reading_ease": 42.5,
        "readability_flesch_kincaid_grade": 14.2,
        "avg_sentence_length": 26.5,
        "passive_voice_pct": 22.0,
        "spelling_mistake_ratio": 0.035,
        "spelling_mistakes": 5.0,
        "lexical_diversity": 0.32,
        "total_links": 0.0,
        "image_count": 0.0,
        "keyword_density": 5.2,
        "transition_word_density": 0.8,
        "sentiment_subjectivity": 0.82
    }
    recommendations = engine.generate_recommendations(dummy_feats)
    print(f"Generated {len(recommendations)} recommendations:")
    for idx, r in enumerate(recommendations[:4]):
        print(f"\n{idx+1}. [{r['category']}] Priority: {r['priority']} - {r['action']}")
        print(f"   Reason: {r['explanation']}")
        print(f"   Impact: {r['impact']}")
