"""
Document Processing Tool for Research Assistant

This module provides document processing capabilities including:
- Text extraction and analysis
- Key information extraction
- Document summarization
- Content categorization
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section of a document."""

    title: str
    content: str
    section_type: str  # introduction, methodology, results, conclusion, etc.
    importance_score: float
    key_phrases: List[str] = None


@dataclass
class DocumentAnalysis:
    """Represents the analysis of a document."""

    document_id: str
    title: str
    summary: str
    key_findings: List[str]
    statistics: List[str]
    citations: List[str]
    sections: List[DocumentSection]
    metadata: Dict[str, Any]
    analysis_timestamp: datetime


class DocumentProcessor:
    """
    Document processing tool for extracting and analyzing content.

    This tool provides:
    - Text extraction and cleaning
    - Key information identification
    - Document summarization
    - Content categorization
    - Citation extraction
    """

    def __init__(self):
        """Initialize the document processor."""
        self.supported_formats = ["txt", "md", "json", "html"]

    def process_text(self, text: str, document_id: str = None) -> DocumentAnalysis:
        """
        Process and analyze text content.

        Args:
            text: The text content to process
            document_id: Optional document identifier

        Returns:
            Document analysis results
        """
        if not document_id:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Clean and preprocess text
        cleaned_text = self._clean_text(text)

        # Extract document sections
        sections = self._extract_sections(cleaned_text)

        # Generate summary
        summary = self._generate_summary(cleaned_text)

        # Extract key findings
        key_findings = self._extract_key_findings(cleaned_text)

        # Extract statistics
        statistics = self._extract_statistics(cleaned_text)

        # Extract citations
        citations = self._extract_citations(cleaned_text)

        # Create analysis
        analysis = DocumentAnalysis(
            document_id=document_id,
            title=self._extract_title(cleaned_text),
            summary=summary,
            key_findings=key_findings,
            statistics=statistics,
            citations=citations,
            sections=sections,
            metadata={
                "word_count": len(cleaned_text.split()),
                "section_count": len(sections),
                "has_statistics": len(statistics) > 0,
                "has_citations": len(citations) > 0,
            },
            analysis_timestamp=datetime.now(),
        )

        logger.info(
            f"Processed document {document_id}: {len(sections)} sections, {len(key_findings)} findings"
        )
        return analysis

    def extract_key_insights(self, documents: List[DocumentAnalysis]) -> Dict[str, Any]:
        """
        Extract key insights across multiple documents.

        Args:
            documents: List of document analyses

        Returns:
            Dictionary containing cross-document insights
        """
        insights = {
            "common_themes": [],
            "contradictions": [],
            "supporting_evidence": [],
            "gaps": [],
            "trends": [],
            "confidence_score": 0.0,
        }

        if not documents:
            return insights

        # Extract common themes
        all_findings = []
        for doc in documents:
            all_findings.extend(doc.key_findings)

        insights["common_themes"] = self._identify_common_themes(all_findings)

        # Identify contradictions
        insights["contradictions"] = self._identify_contradictions(documents)

        # Collect supporting evidence
        insights["supporting_evidence"] = self._collect_supporting_evidence(documents)

        # Identify research gaps
        insights["gaps"] = self._identify_research_gaps(documents)

        # Calculate confidence score
        insights["confidence_score"] = self._calculate_confidence_score(documents)

        return insights

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep important punctuation
        text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]", "", text)

        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        return text.strip()

    def _extract_sections(self, text: str) -> List[DocumentSection]:
        """Extract document sections based on headers and structure."""
        sections = []

        # Split by common section markers
        section_patterns = [
            r"(?:^|\n)([A-Z][A-Z\s]+:?)",  # ALL CAPS headers
            r"(?:^|\n)(\d+\.\s+[A-Z][^:\n]+)",  # Numbered sections
            r"(?:^|\n)([A-Z][^:\n]{3,50}:?)",  # Title case headers
        ]

        lines = text.split("\n")
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line is a section header
            is_header = False
            section_type = "content"

            for pattern in section_patterns:
                if re.match(pattern, line):
                    is_header = True
                    if "introduction" in line.lower() or "overview" in line.lower():
                        section_type = "introduction"
                    elif "method" in line.lower() or "approach" in line.lower():
                        section_type = "methodology"
                    elif "result" in line.lower() or "finding" in line.lower():
                        section_type = "results"
                    elif "conclusion" in line.lower() or "summary" in line.lower():
                        section_type = "conclusion"
                    break

            if is_header and current_section:
                # Save previous section
                if current_content:
                    section = DocumentSection(
                        title=current_section,
                        content="\n".join(current_content),
                        section_type=section_type,
                        importance_score=self._calculate_section_importance(
                            current_content
                        ),
                        key_phrases=self._extract_key_phrases(current_content),
                    )
                    sections.append(section)

                # Start new section
                current_section = line
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
                else:
                    # First line becomes title if no header found
                    current_section = "Introduction"
                    current_content.append(line)

        # Add final section
        if current_section and current_content:
            section = DocumentSection(
                title=current_section,
                content="\n".join(current_content),
                section_type="content",
                importance_score=self._calculate_section_importance(current_content),
                key_phrases=self._extract_key_phrases(current_content),
            )
            sections.append(section)

        return sections

    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the document."""
        # Simple extractive summarization
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Score sentences by importance (simple heuristic)
        sentence_scores = []
        for sentence in sentences:
            score = 0
            # Score based on length
            score += min(len(sentence.split()), 30) / 30

            # Score based on keywords
            keywords = [
                "important",
                "significant",
                "key",
                "major",
                "finding",
                "result",
                "conclusion",
            ]
            score += sum(1 for keyword in keywords if keyword in sentence.lower())

            # Score based on numbers (statistics)
            if re.search(r"\d+", sentence):
                score += 2

            sentence_scores.append((sentence, score))

        # Select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sentence_scores[:3]

        summary = ". ".join([s[0] for s in top_sentences]) + "."
        return summary

    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from the text."""
        findings = []

        # Look for sentences with finding indicators
        finding_patterns = [
            r"[^.]*(?:finding|discovered|revealed|showed|demonstrated|indicated)[^.]*\.",
            r"[^.]*(?:study|research|analysis|investigation)\s+(?:found|revealed|showed)[^.]*\.",
            r"[^.]*(?:result|outcome|conclusion)\s+(?:was|is|shows)[^.]*\.",
        ]

        for pattern in finding_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            findings.extend(matches)

        # Clean and deduplicate findings
        findings = [f.strip() for f in findings if len(f.strip()) > 20]
        findings = list(set(findings))

        return findings[:10]  # Limit to top 10 findings

    def _extract_statistics(self, text: str) -> List[str]:
        """Extract statistical information from the text."""
        statistics = []

        # Look for statistical patterns
        stat_patterns = [
            r"\d+(?:\.\d+)?\s*(?:percent|%|per cent)",
            r"(?:increased|decreased|grew|declined)\s+by\s+\d+(?:\.\d+)?\s*(?:percent|%|per cent)",
            r"\d+(?:\.\d+)?\s*(?:million|billion|thousand)",
            r"(?:average|mean|median)\s+(?:of|was)\s+\d+(?:\.\d+)?",
            r"\d+(?:\.\d+)?\s*(?:times|fold)\s+(?:higher|lower|more|less)",
        ]

        for pattern in stat_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            statistics.extend(matches)

        return list(set(statistics))[:10]

    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations and references from the text."""
        citations = []

        # Look for citation patterns
        citation_patterns = [
            r"\([^)]*\d{4}[^)]*\)",  # (Author, 2024)
            r"\[[^\]]*\d{4}[^\]]*\]",  # [Author, 2024]
            r"(?:et al\.|and colleagues|and others)\s*\(\d{4}\)",
            r"(?:cited in|referenced in|according to)\s+[^,\.]+",
        ]

        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)

        return list(set(citations))[:10]

    def _extract_title(self, text: str) -> str:
        """Extract document title."""
        lines = text.split("\n")
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 100 and not line.endswith("."):
                return line
        return "Untitled Document"

    def _calculate_section_importance(self, content: List[str]) -> float:
        """Calculate importance score for a section."""
        if not content:
            return 0.0

        text = " ".join(content)
        score = 0.0

        # Score based on length
        score += min(len(text.split()), 200) / 200

        # Score based on keywords
        important_words = [
            "important",
            "significant",
            "key",
            "major",
            "finding",
            "result",
            "conclusion",
            "evidence",
        ]
        score += sum(1 for word in important_words if word in text.lower()) / len(
            important_words
        )

        # Score based on numbers (statistics)
        if re.search(r"\d+", text):
            score += 0.3

        return min(score, 1.0)

    def _extract_key_phrases(self, content: List[str]) -> List[str]:
        """Extract key phrases from content."""
        text = " ".join(content)

        # Simple key phrase extraction
        phrases = []

        # Look for noun phrases
        noun_patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",  # Title case phrases
            r"\b\w+(?:\s+\w+){2,4}\b",  # Multi-word phrases
        ]

        for pattern in noun_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)

        # Filter and limit phrases
        phrases = [p for p in phrases if len(p.split()) >= 2 and len(p) > 5]
        return list(set(phrases))[:10]

    def _identify_common_themes(self, findings: List[str]) -> List[str]:
        """Identify common themes across findings."""
        # Simple theme identification based on common words
        all_text = " ".join(findings).lower()

        # Count word frequencies
        words = re.findall(r"\b\w{4,}\b", all_text)
        word_counts = {}
        for word in words:
            if word not in [
                "this",
                "that",
                "with",
                "from",
                "they",
                "have",
                "been",
                "were",
                "will",
                "said",
            ]:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Get most common words as themes
        themes = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [theme[0] for theme in themes]

    def _identify_contradictions(self, documents: List[DocumentAnalysis]) -> List[str]:
        """Identify contradictions across documents."""
        contradictions = []

        # Simple contradiction detection
        all_findings = []
        for doc in documents:
            all_findings.extend(doc.key_findings)

        # Look for opposing statements
        positive_words = ["increase", "growth", "improve", "positive", "success"]
        negative_words = ["decrease", "decline", "worse", "negative", "failure"]

        for i, finding1 in enumerate(all_findings):
            for finding2 in all_findings[i + 1 :]:
                has_positive = any(word in finding1.lower() for word in positive_words)
                has_negative = any(word in finding2.lower() for word in negative_words)

                if has_positive and has_negative:
                    contradictions.append(
                        f"Contradiction between findings: {finding1[:50]}... vs {finding2[:50]}..."
                    )

        return contradictions[:5]

    def _collect_supporting_evidence(
        self, documents: List[DocumentAnalysis]
    ) -> List[str]:
        """Collect supporting evidence across documents."""
        evidence = []

        for doc in documents:
            evidence.extend(doc.statistics)
            evidence.extend(doc.key_findings[:3])  # Top 3 findings per document

        return evidence[:10]

    def _identify_research_gaps(self, documents: List[DocumentAnalysis]) -> List[str]:
        """Identify research gaps based on document analysis."""
        gaps = []

        # Look for gap indicators in text
        gap_indicators = [
            "further research needed",
            "more studies required",
            "limited data available",
            "insufficient evidence",
            "gaps in knowledge",
            "future research should",
            "additional investigation needed",
        ]

        for doc in documents:
            text = " ".join([s.content for s in doc.sections])
            for indicator in gap_indicators:
                if indicator in text.lower():
                    gaps.append(f"Research gap identified in {doc.title}: {indicator}")

        return gaps[:5]

    def _calculate_confidence_score(self, documents: List[DocumentAnalysis]) -> float:
        """Calculate confidence score based on document quality."""
        if not documents:
            return 0.0

        total_score = 0.0

        for doc in documents:
            # Score based on document completeness
            score = 0.0

            # Has summary
            if doc.summary:
                score += 0.2

            # Has findings
            if doc.key_findings:
                score += 0.2

            # Has statistics
            if doc.statistics:
                score += 0.2

            # Has citations
            if doc.citations:
                score += 0.2

            # Has multiple sections
            if len(doc.sections) > 2:
                score += 0.2

            total_score += score

        return total_score / len(documents)
