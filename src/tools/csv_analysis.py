"""
CSV Analysis Tool for Research Assistant

This module provides tabular data analysis capabilities using pandas
to perform basic statistical analysis on CSV datasets and translate
findings into natural language summaries.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from src.models.model_builder import ModelBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Represents the result of a CSV analysis."""

    dataset_name: str
    dataset_description: str
    statistics: Dict[str, Any]
    summary: str
    relevant_columns: List[str]
    key_insights: List[str]
    sample_data: Dict[str, Any]
    metadata: Dict[str, Any]


class CSVAnalysisTool:
    """
    Tool for performing basic statistical analysis on CSV datasets
    and providing natural language summaries of findings.
    """

    def __init__(self, model_builder: Optional[ModelBuilder] = None):
        """
        Initialize the CSV Analysis Tool.

        Args:
            model_builder: Optional ModelBuilder instance for AI-powered summaries
        """
        self.model_builder = model_builder or ModelBuilder()
        # Try to build the model, but handle gracefully if no API key
        self.model = None
        self.ai_enabled = False

        try:
            self.model = self.model_builder.with_provider("openai").build()
            self.ai_enabled = True
            logger.info("AI-powered summaries enabled")
        except Exception as e:
            logger.warning(f"AI model not available: {e}. Using basic summaries only.")
            self.ai_enabled = False

        # Paths for CSV data
        self.csv_dir = (
            "/home/vlofgren/Documents/Projects/research-assistant-mcp/data/csvs"
        )
        self.descriptions_file = "/home/vlofgren/Documents/Projects/research-assistant-mcp/data/csv_descriptions.csv"

        # Load dataset descriptions
        self.dataset_descriptions = self._load_descriptions()

    def _load_descriptions(self) -> Dict[str, str]:
        """Load CSV descriptions from the descriptions file."""
        try:
            if os.path.exists(self.descriptions_file):
                desc_df = pd.read_csv(self.descriptions_file)
                return dict(zip(desc_df["filename"], desc_df["description"]))
            else:
                logger.warning(f"Descriptions file not found: {self.descriptions_file}")
                return {}
        except Exception as e:
            logger.error(f"Error loading descriptions: {e}")
            return {}

    def _find_relevant_datasets(self, query: str) -> List[Tuple[str, str]]:
        """
        Find datasets relevant to the query based on descriptions.

        Args:
            query: Search query or research question

        Returns:
            List of (filename, description) tuples for relevant datasets
        """
        relevant = []
        query_lower = query.lower()

        for filename, description in self.dataset_descriptions.items():
            if any(word in description.lower() for word in query_lower.split()):
                relevant.append((filename, description))

        # If no matches found, return all datasets
        if not relevant:
            relevant = list(self.dataset_descriptions.items())

        return relevant[:3]  # Limit to top 3 most relevant

    def _load_csv_safely(
        self, filepath: str, max_rows: int = 10000
    ) -> Optional[pd.DataFrame]:
        """
        Safely load a CSV file with error handling and size limits.

        Args:
            filepath: Path to the CSV file
            max_rows: Maximum number of rows to load

        Returns:
            DataFrame or None if loading fails
        """
        encodings_to_try = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
        separators_to_try = [",", ";", "\t", "|"]

        for encoding in encodings_to_try:
            for sep in separators_to_try:
                try:
                    # First, try to read just the header to check structure
                    sample = pd.read_csv(
                        filepath,
                        nrows=5,
                        encoding=encoding,
                        sep=sep,
                        on_bad_lines="skip",
                        low_memory=False,
                    )

                    if len(sample.columns) > 1:  # Valid CSV structure
                        logger.info(
                            f"CSV structure preview for {os.path.basename(filepath)} (encoding: {encoding}, sep: '{sep}'): {sample.columns.tolist()}"
                        )

                        # Load full dataset with row limit
                        df = pd.read_csv(
                            filepath,
                            nrows=max_rows,
                            encoding=encoding,
                            sep=sep,
                            on_bad_lines="skip",
                            low_memory=False,
                        )
                        logger.info(
                            f"Loaded {len(df)} rows from {os.path.basename(filepath)} with {encoding} encoding and '{sep}' separator"
                        )
                        return df

                except UnicodeDecodeError:
                    logger.debug(
                        f"Failed to load {os.path.basename(filepath)} with {encoding} encoding"
                    )
                    break  # Try next encoding
                except Exception as e:
                    logger.debug(
                        f"Failed with {encoding} encoding and '{sep}' separator: {e}"
                    )
                    continue

        logger.error(f"Failed to load {filepath} with any encoding")
        return None

    def _perform_basic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform basic statistical analysis on a DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()
            analysis["correlations"] = (
                df[numeric_cols].corr().to_dict() if len(numeric_cols) > 1 else {}
            )

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if categorical_cols:
            analysis["categorical_summary"] = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                value_counts = df[col].value_counts().head(10)
                analysis["categorical_summary"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_values": value_counts.to_dict(),
                    "most_frequent": (
                        str(value_counts.index[0]) if len(value_counts) > 0 else None
                    ),
                }

        # Date columns (if any)
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if date_cols:
            analysis["date_summary"] = {}
            for col in date_cols:
                analysis["date_summary"][col] = {
                    "min_date": str(df[col].min()),
                    "max_date": str(df[col].max()),
                    "date_range_days": (df[col].max() - df[col].min()).days,
                }

        return analysis

    def _extract_key_insights(
        self, df: pd.DataFrame, analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Extract key insights from the analysis results.

        Args:
            df: Original DataFrame
            analysis: Analysis results dictionary

        Returns:
            List of key insight strings
        """
        insights = []

        # Dataset size insights
        rows, cols = analysis["shape"]["rows"], analysis["shape"]["columns"]
        insights.append(f"Dataset contains {rows:,} rows and {cols} columns")

        # Missing data insights
        missing_data = analysis["missing_values"]
        total_missing = sum(missing_data.values())
        if total_missing > 0:
            missing_pct = (total_missing / (rows * cols)) * 100
            insights.append(
                f"Dataset has {total_missing:,} missing values ({missing_pct:.1f}% of total data)"
            )

        # Numeric insights
        if "numeric_summary" in analysis:
            numeric_cols = list(analysis["numeric_summary"].keys())
            insights.append(
                f"Contains {len(numeric_cols)} numeric columns: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}"
            )

            # Check for potential outliers or interesting patterns
            for col in numeric_cols[:3]:
                stats = analysis["numeric_summary"][col]
                if stats["std"] > 0:
                    cv = (
                        stats["std"] / abs(stats["mean"])
                        if stats["mean"] != 0
                        else float("inf")
                    )
                    if cv > 1:
                        insights.append(
                            f"Column '{col}' shows high variability (CV: {cv:.2f})"
                        )

        # Categorical insights
        if "categorical_summary" in analysis:
            cat_summary = analysis["categorical_summary"]
            for col, info in list(cat_summary.items())[:2]:
                unique_count = info["unique_values"]
                insights.append(f"Column '{col}' has {unique_count:,} unique values")
                if info["most_frequent"]:
                    insights.append(
                        f"Most frequent value in '{col}': {info['most_frequent']}"
                    )

        return insights

    def _generate_summary(
        self,
        dataset_name: str,
        query: str,
        analysis: Dict[str, Any],
        insights: List[str],
    ) -> str:
        """
        Generate a natural language summary of the analysis using AI.

        Args:
            dataset_name: Name of the dataset
            query: Original query/question
            analysis: Analysis results
            insights: Key insights list

        Returns:
            Natural language summary
        """
        summary_prompt = f"""
        You are analyzing a dataset called "{dataset_name}" in response to the query: "{query}"
        
        Dataset Analysis Results:
        {json.dumps(analysis, indent=2, default=str)}
        
        Key Insights:
        {chr(10).join(f"- {insight}" for insight in insights)}
        
        Please provide a comprehensive but concise summary that:
        1. Explains what this dataset contains and its relevance to the query
        2. Highlights the most important statistical findings
        3. Identifies any notable patterns, trends, or anomalies
        4. Suggests what questions this data might help answer
        5. Notes any limitations or data quality issues
        
        Write in clear, accessible language suitable for researchers.
        """

        if self.ai_enabled:
            try:
                summary = self.model.run(summary_prompt)
                return summary
            except Exception as e:
                logger.error(f"Failed to generate AI summary: {e}")

        # Fallback to basic summary
        basic_summary = f"""Analysis of {dataset_name}:
- Dataset contains {analysis['shape']['rows']:,} rows and {analysis['shape']['columns']} columns
- Key insights: {'; '.join(insights[:3])}
- This dataset appears relevant to queries about: {query}

Statistical Summary:
"""

        # Add numeric column summaries
        if "numeric_summary" in analysis and analysis["numeric_summary"]:
            basic_summary += "- Numeric columns: "
            for col, stats in list(analysis["numeric_summary"].items())[:3]:
                basic_summary += f"{col} (mean: {stats.get('mean', 0):.2f}, std: {stats.get('std', 0):.2f}); "
            basic_summary = basic_summary.rstrip("; ")
            basic_summary += "\n"

        # Add categorical column summaries
        if "categorical_summary" in analysis and analysis["categorical_summary"]:
            basic_summary += "- Categorical columns: "
            for col, info in list(analysis["categorical_summary"].items())[:2]:
                basic_summary += f"{col} ({info['unique_values']} unique values); "
            basic_summary = basic_summary.rstrip("; ")
            basic_summary += "\n"

        return basic_summary

    def analyze_for_query(self, query: str, max_datasets: int = 2) -> Dict[str, Any]:
        """
        Analyze relevant CSV datasets based on a research query.

        Args:
            query: Research question or query
            max_datasets: Maximum number of datasets to analyze

        Returns:
            Dictionary containing analysis results for all relevant datasets
        """
        logger.info(f"Starting CSV analysis for query: {query}")

        # Find relevant datasets
        relevant_datasets = self._find_relevant_datasets(query)

        if not relevant_datasets:
            return {
                "query": query,
                "datasets_analyzed": 0,
                "results": [],
                "summary": "No relevant CSV datasets found for the query.",
                "processing_details": {
                    "datasets_available": len(self.dataset_descriptions),
                    "datasets_matched": 0,
                    "analysis_timestamp": datetime.now().isoformat(),
                },
            }

        results = []
        datasets_processed = 0

        for filename, description in relevant_datasets[:max_datasets]:
            filepath = os.path.join(self.csv_dir, filename)

            if not os.path.exists(filepath):
                logger.warning(f"Dataset file not found: {filepath}")
                continue

            # Load and analyze dataset
            df = self._load_csv_safely(filepath)
            if df is None:
                continue

            try:
                # Perform analysis
                analysis = self._perform_basic_analysis(df)
                insights = self._extract_key_insights(df, analysis)
                summary = self._generate_summary(filename, query, analysis, insights)

                # Create sample data
                sample_data = {
                    "first_few_rows": (
                        df.head(3).to_dict("records") if len(df) > 0 else []
                    ),
                    "column_names": df.columns.tolist(),
                    "shape": analysis["shape"],
                }

                # Create result object
                result = AnalysisResult(
                    dataset_name=filename,
                    dataset_description=description,
                    statistics=analysis,
                    summary=summary,
                    relevant_columns=df.columns.tolist()[
                        :10
                    ],  # Limit to first 10 columns
                    key_insights=insights,
                    sample_data=sample_data,
                    metadata={
                        "file_size_mb": os.path.getsize(filepath) / (1024 * 1024),
                        "analysis_timestamp": datetime.now().isoformat(),
                    },
                )

                results.append(
                    {
                        "dataset": result.dataset_name,
                        "description": result.dataset_description,
                        "summary": result.summary,
                        "key_insights": result.key_insights,
                        "statistics": {
                            "shape": result.statistics["shape"],
                            "columns": result.relevant_columns,
                            "data_types": len(
                                set(result.statistics["data_types"].values())
                            ),
                            "missing_data_pct": sum(
                                result.statistics["missing_values"].values()
                            )
                            / (
                                result.statistics["shape"]["rows"]
                                * result.statistics["shape"]["columns"]
                            )
                            * 100,
                        },
                        "sample_data": result.sample_data,
                        "type": "tabular_analysis",
                    }
                )

                datasets_processed += 1
                logger.info(f"Successfully analyzed dataset: {filename}")

            except Exception as e:
                logger.error(f"Error analyzing {filename}: {e}")
                continue

        # Generate overall summary
        overall_summary = self._generate_overall_summary(query, results)

        return {
            "query": query,
            "datasets_analyzed": datasets_processed,
            "results": results,
            "summary": overall_summary,
            "processing_details": {
                "datasets_available": len(self.dataset_descriptions),
                "datasets_matched": len(relevant_datasets),
                "datasets_processed": datasets_processed,
                "analysis_timestamp": datetime.now().isoformat(),
            },
        }

    def _generate_overall_summary(
        self, query: str, results: List[Dict[str, Any]]
    ) -> str:
        """Generate an overall summary across all analyzed datasets."""
        if not results:
            return "No datasets were successfully analyzed."

        datasets_summary = []
        for result in results:
            datasets_summary.append(f"- {result['dataset']}: {result['description']}")

        summary_prompt = f"""
        Based on analysis of {len(results)} datasets in response to the query "{query}":
        
        Datasets analyzed:
        {chr(10).join(datasets_summary)}
        
        Key findings across datasets:
        {chr(10).join([insight for result in results for insight in result['key_insights'][:2]])}
        
        Provide a concise overall summary that:
        1. Explains how these datasets collectively address the research query
        2. Highlights the most significant findings across all datasets
        3. Identifies connections or patterns between datasets
        4. Suggests follow-up questions or additional analysis needed
        """

        if self.ai_enabled:
            try:
                return self.model.run(summary_prompt)
            except Exception as e:
                logger.error(f"Failed to generate overall summary: {e}")

        # Fallback to basic overall summary
        summary = f"Analyzed {len(results)} datasets relevant to: {query}\n\n"

        for i, result in enumerate(results, 1):
            summary += f"{i}. {result['dataset']}: {result['description']}\n"
            summary += f"   - {result['statistics']['shape']['rows']:,} rows, {result['statistics']['shape']['columns']} columns\n"
            if result["key_insights"]:
                summary += f"   - Key finding: {result['key_insights'][0]}\n"
            summary += "\n"

        summary += (
            "Each dataset provides complementary insights for understanding this topic."
        )
        return summary


# Function to test CSV analysis
def test_csv_analysis(query: str) -> Dict[str, Any]:
    """
    Perform a quick CSV analysis test.

    Args:
        query: The research query

    Returns:
        Analysis results
    """
    analysis_tool = CSVAnalysisTool()
    return analysis_tool.analyze_for_query(query)
