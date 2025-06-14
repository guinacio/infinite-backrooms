#!/usr/bin/env python3
"""
AI Conversation Log Viewer - Streamlit App
"""

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional
import glob

class LogParser:
    """Parser for AI conversation log files"""
    
    def __init__(self, log_dir: str = "conversations"):
        self.log_dir = Path(log_dir)
    
    def get_available_log_files(self) -> List[Path]:
        """Get all available log files"""
        if not self.log_dir.exists():
            return []
        
        log_files = []
        # Look for both old backroom format and new ai_conversation format
        patterns = ["backroom_*.txt", "ai_conversation_*.txt"]
        
        for pattern in patterns:
            log_files.extend(self.log_dir.glob(pattern))
        
        return sorted(log_files, reverse=True)  # Most recent first
    
    def parse_log_file(self, file_path: Path) -> List[Dict]:
        """Parse a single log file into structured data"""
        messages = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse format: [HH:MM:SS] Persona$ Message
                    match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s+([^$]+)\$\s+(.*)', line)
                    if match:
                        timestamp_str, persona, message = match.groups()
                        
                        # Extract date from filename
                        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path.name)
                        if date_match:
                            date_str = date_match.group(1)
                            full_timestamp = f"{date_str} {timestamp_str}"
                        else:
                            full_timestamp = timestamp_str
                        
                        messages.append({
                            'timestamp': timestamp_str,
                            'full_timestamp': full_timestamp,
                            'persona': persona.strip(),
                            'message': message.strip(),
                            'file': file_path.name,
                            'line_number': line_num,
                            'message_length': len(message.strip())
                        })
        
        except Exception as e:
            st.error(f"Error parsing {file_path.name}: {str(e)}")
        
        return messages
    
    def parse_all_logs(self, selected_files: List[Path] = None) -> pd.DataFrame:
        """Parse all selected log files into a DataFrame"""
        all_messages = []
        
        files_to_parse = selected_files if selected_files else self.get_available_log_files()
        
        for file_path in files_to_parse:
            messages = self.parse_log_file(file_path)
            all_messages.extend(messages)
        
        if not all_messages:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_messages)
        
        # Convert timestamp to datetime for better sorting/filtering
        try:
            df['datetime'] = pd.to_datetime(df['full_timestamp'])
        except:
            df['datetime'] = pd.NaT
        
        return df.sort_values('datetime', ascending=False).reset_index(drop=True)


def create_sidebar_filters(df: pd.DataFrame) -> Tuple[List[str], str, str, date, date]:
    """Create sidebar filters and return selected values"""
    st.sidebar.header("🔍 Filters")
    
    # Persona filter
    if not df.empty:
        available_personas = sorted(df['persona'].unique())
        selected_personas = st.sidebar.multiselect(
            "Select Personas",
            options=available_personas,
            default=available_personas,
            help="Filter messages by AI persona"
        )
    else:
        selected_personas = []
    
    # Search term
    search_term = st.sidebar.text_input(
        "Search Messages",
        placeholder="Enter keywords to search...",
        help="Search within message content (case-insensitive)"
    )
    
    # Show search debug info when search is active
    if search_term:
        st.sidebar.caption(f"🔍 Searching for: '{search_term}'")
    
    # Search type
    search_type = st.sidebar.radio(
        "Search Type",
        options=["Contains", "Exact Match", "Regex"],
        help="How to match the search term"
    )
    
    # Date range filter
    if not df.empty and 'datetime' in df.columns:
        min_date = df['datetime'].dt.date.min()
        max_date = df['datetime'].dt.date.max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            start_date = st.sidebar.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
            
            end_date = st.sidebar.date_input(
                "End Date", 
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        else:
            start_date = date.today()
            end_date = date.today()
    else:
        start_date = date.today()
        end_date = date.today()
    
    return selected_personas, search_term, search_type, start_date, end_date


def apply_filters(df: pd.DataFrame, personas: List[str], search_term: str, 
                 search_type: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Apply all filters to the DataFrame"""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Persona filter
    if personas:
        filtered_df = filtered_df[filtered_df['persona'].isin(personas)]
    
    # Date filter
    if 'datetime' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['datetime'].dt.date >= start_date) &
            (filtered_df['datetime'].dt.date <= end_date)
        ]
    
    # Search filter - fixed the search functionality
    if search_term:
        if search_type == "Contains":
            mask = filtered_df['message'].str.contains(search_term, case=False, na=False)
        elif search_type == "Exact Match":
            # For exact match, we don't need regex, just direct string comparison
            mask = filtered_df['message'].str.lower().str.contains(search_term.lower(), regex=False)
        elif search_type == "Regex":
            try:
                mask = filtered_df['message'].str.contains(search_term, case=False, na=False, regex=True)
            except re.error as e:
                st.error(f"Invalid regex pattern: {str(e)}")
                mask = pd.Series([False] * len(filtered_df))
        
        filtered_df = filtered_df[mask]
    
    return filtered_df


def display_statistics(df: pd.DataFrame):
    """Display conversation statistics"""
    if df.empty:
        st.info("No messages to display")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", len(df))
    
    with col2:
        unique_personas = df['persona'].nunique()
        st.metric("Active Personas", unique_personas)
    
    with col3:
        avg_length = df['message_length'].mean()
        st.metric("Avg Message Length", f"{avg_length:.0f} chars")
    
    with col4:
        if 'datetime' in df.columns:
            date_range = df['datetime'].dt.date.max() - df['datetime'].dt.date.min()
            st.metric("Date Range", f"{date_range.days + 1} days")
        else:
            st.metric("Files", df['file'].nunique())


def display_persona_breakdown(df: pd.DataFrame):
    """Display breakdown by persona"""
    if df.empty:
        return
    
    st.subheader("📊 Persona Activity")
    
    persona_stats = df.groupby('persona').agg({
        'message': 'count',
        'message_length': ['mean', 'sum']
    }).round(1)
    
    persona_stats.columns = ['Message Count', 'Avg Length', 'Total Chars']
    persona_stats = persona_stats.sort_values('Message Count', ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(persona_stats, use_container_width=True)
    
    with col2:
        # Create a simple bar chart
        chart_data = persona_stats['Message Count'].reset_index()
        st.bar_chart(chart_data.set_index('persona'))


def display_messages(df: pd.DataFrame):
    """Display the filtered messages"""
    if df.empty:
        st.info("No messages match the current filters")
        return
    
    st.subheader(f"💬 Messages ({len(df)})")
    
    # Display format options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_format = st.radio(
            "Display Format",
            options=["Chat View", "Table View", "Raw Text"],
            horizontal=True
        )
    
    # Sort options for Chat View
    sort_order = "newest"
    if display_format == "Chat View":
        with col2:
            sort_order = st.selectbox(
                "Sort Order",
                options=["newest", "oldest"],
                help="Sort messages by timestamp"
            )
    
    # Apply sorting for Chat View
    if display_format == "Chat View":
        if sort_order == "newest":
            display_df = df.sort_values('datetime', ascending=False) if 'datetime' in df.columns else df
        else:  # oldest
            display_df = df.sort_values('datetime', ascending=True) if 'datetime' in df.columns else df
    else:
        display_df = df
    
    if display_format == "Chat View":
        for _, row in display_df.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 6])
                
                with col1:
                    st.write(f"**{row['persona']}**")
                    st.caption(f"{row['timestamp']}")
                
                with col2:
                    st.write(row['message'])
                
                st.divider()
    
    elif display_format == "Table View":
        # Select columns to display
        display_cols = st.multiselect(
            "Select columns to display",
            options=['timestamp', 'persona', 'message', 'file', 'message_length'],
            default=['timestamp', 'persona', 'message']
        )
        
        if display_cols:
            st.dataframe(
                display_df[display_cols],
                use_container_width=True,
                hide_index=True
            )
    
    elif display_format == "Raw Text":
        # Show raw text format
        raw_text = ""
        for _, row in display_df.iterrows():
            raw_text += f"[{row['timestamp']}] {row['persona']}$ {row['message']}\n"
        
        st.text_area(
            "Raw Log Format",
            value=raw_text,
            height=400,
            help="Copy-pasteable raw log format"
        )


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="AI Conversation Log Viewer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Conversation Log Viewer")
    st.markdown("View and analyze AI conversation logs with filtering and search capabilities")
    
    # Initialize parser
    parser = LogParser()
    
    # File selection
    available_files = parser.get_available_log_files()
    
    if not available_files:
        st.error("No log files found in the 'conversations' directory!")
        st.info("Make sure you have run the AI conversation script to generate log files.")
        return
    
    st.sidebar.header("📁 Log Files")
    selected_files = st.sidebar.multiselect(
        "Select log files to analyze",
        options=available_files,
        default=available_files[:3],  # Default to 3 most recent
        format_func=lambda x: x.name
    )
    
    if not selected_files:
        st.warning("Please select at least one log file to analyze")
        return
    
    # Parse logs
    with st.spinner("Loading and parsing log files..."):
        df = parser.parse_all_logs(selected_files)
    
    # Create filters
    personas, search_term, search_type, start_date, end_date = create_sidebar_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, personas, search_term, search_type, start_date, end_date)
    
    # Show search results summary
    if search_term and not df.empty:
        original_count = len(df)
        filtered_count = len(filtered_df)
        if filtered_count != original_count:
            st.info(f"🔍 Search results: Found {filtered_count} messages containing '{search_term}' out of {original_count} total messages")
        elif filtered_count == 0:
            st.warning(f"🔍 No messages found containing '{search_term}'")
    
    # Display statistics
    display_statistics(filtered_df)
    
    # Display persona breakdown
    if not filtered_df.empty:
        display_persona_breakdown(filtered_df)
    
    # Display messages
    st.markdown("---")
    display_messages(filtered_df)
    
    # Export functionality
    if not filtered_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.header("💾 Export")
        
        export_format = st.sidebar.selectbox(
            "Export Format",
            options=["CSV", "JSON", "TXT"]
        )
        
        if st.sidebar.button("Download Filtered Data"):
            if export_format == "CSV":
                csv = filtered_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"ai_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif export_format == "JSON":
                json_data = filtered_df.to_json(orient='records', indent=2)
                st.sidebar.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"ai_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            elif export_format == "TXT":
                txt_data = ""
                for _, row in filtered_df.iterrows():
                    txt_data += f"[{row['timestamp']}] {row['persona']}$ {row['message']}\n"
                
                st.sidebar.download_button(
                    label="📥 Download TXT",
                    data=txt_data,
                    file_name=f"ai_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )


if __name__ == "__main__":
    main() 