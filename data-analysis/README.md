# 🚶 Locomotion Data Analysis Dashboard

An interactive web-based dashboard for analyzing locomotion command data with comprehensive visualizations and filtering capabilities.

## 📊 Features

### 📈 Interactive Visualizations
- **Locomotion Modes Distribution**: Doughnut chart showing the distribution of different locomotion modes
- **Command Types Distribution**: Bar chart displaying Clear, Vague, and Safety-Critical command types
- **Timeline of Activities**: Line chart showing command frequency over time
- **Duration Analysis**: Histogram of command durations

### 🔍 Advanced Filtering
- **Locomotion Mode Filter**: Filter by specific locomotion modes (Level-Ground Navigation, Vertical Ladder Climbing, etc.)
- **Command Type Filter**: Filter by command clarity (Clear, Vague, Safety-Critical)
- **Time Range Filter**: Filter commands within specific time ranges
- **Text Search**: Search through command descriptions

### 📋 Data Views
- **Statistics Cards**: Key metrics including total commands, unique modes, command types, and average duration
- **Activity Timeline**: Chronological view of all commands with color-coded modes and types
- **Detailed Data Table**: Paginated table with all command details and calculated durations

### 🎨 Modern UI
- Responsive design that works on desktop and mobile devices
- Color-coded locomotion modes and command types
- Smooth animations and hover effects
- Professional gradient backgrounds and card-based layout

## 🚀 Getting Started

### Prerequisites
- Python 3.6 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation & Usage

1. **Navigate to the data-analysis folder**
   ```bash
   cd data-analysis
   ```

2. **Start the local server**
   ```bash
   python server.py
   ```

3. **Open your browser**
   Navigate to: `http://localhost:8000`

4. **Explore the dashboard**
   - Use the filters at the top to narrow down your data
   - Click on chart elements to see detailed information
   - Scroll through the timeline and data table
   - Use the search box to find specific commands

## 📁 File Structure

```
data-analysis/
├── index.html          # Main dashboard HTML file
├── server.py           # Python HTTP server
├── README.md           # This file
├── data/               # Symlink to ../data (locomotion command data)
│   └── data.json      # JSON data file
└── images/             # Symlink to ../images (image files)
```

## 📊 Data Format

The dashboard expects JSON data in the following format:

```json
[
  {
    "id": "unique-identifier",
    "timestamp_start": "HH:MM:SS.mmm",
    "timestamp_end": "HH:MM:SS.mmm",
    "command": "Description of the command",
    "command_type": "Clear|Vague|Safety-Critical",
    "locomotion_mode": "Specific locomotion mode"
  }
]
```

### Supported Locomotion Modes
- Level-Ground Navigation
- Vertical Ladder Up Climbing
- Vertical Ladder Down Climbing
- Construction Ladder Up Climbing
- Construction Ladder Down Climbing
- Stair Ascension
- Stair Descension
- Stepping over Pipe
- Stepping over Box
- Low Space Navigation
- Sitting Down
- Standing Up

### Command Types
- **Clear**: Well-defined, unambiguous commands
- **Vague**: Ambiguous or unclear commands
- **Safety-Critical**: Commands that require special attention for safety

## 🎯 Use Cases

### For Researchers
- Analyze patterns in locomotion commands
- Identify common command types and modes
- Study temporal distribution of activities
- Export filtered data for further analysis

### For Developers
- Debug locomotion command recognition
- Validate command classification accuracy
- Monitor system performance over time
- Identify edge cases and anomalies

### For Safety Analysis
- Focus on Safety-Critical commands
- Analyze command clarity patterns
- Identify potential safety concerns
- Track command execution durations

## 🔧 Customization

### Adding New Visualizations
The dashboard uses Chart.js for visualizations. You can add new charts by:

1. Adding a new canvas element in the HTML
2. Creating a new chart function in the JavaScript
3. Calling the function in `updateCharts()`

### Modifying Filters
To add new filter options:

1. Add new filter elements in the HTML controls section
2. Update the `applyFilters()` function to handle the new filter
3. Add the filter to the `resetFilters()` function

### Styling Changes
The dashboard uses CSS Grid and Flexbox for layout. You can modify the styles in the `<style>` section of `index.html` to customize the appearance.

## 🐛 Troubleshooting

### Common Issues

**Dashboard doesn't load data**
- Make sure `data/data.json` exists and is valid JSON
- Check the browser console for JavaScript errors
- Verify the server is running on the correct port

**Charts don't display**
- Ensure you have an internet connection (for CDN libraries)
- Check that Chart.js is loading properly
- Verify the canvas elements exist in the HTML

**Filters don't work**
- Check that the filter event listeners are properly attached
- Verify the data structure matches the expected format
- Look for JavaScript errors in the browser console

### Performance Tips
- For large datasets (>1000 records), consider implementing virtual scrolling
- Use the filters to reduce the amount of data being processed
- Consider paginating the timeline view for better performance

## 📈 Future Enhancements

Potential improvements for the dashboard:

- **Export functionality**: Download filtered data as CSV/JSON
- **Advanced analytics**: Statistical analysis and trend detection
- **Real-time updates**: WebSocket integration for live data
- **User preferences**: Save filter settings and chart configurations
- **Image integration**: Display related images alongside commands
- **Collaborative features**: Share analysis results and annotations

## 🤝 Contributing

To contribute to this dashboard:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use and modify for your own locomotion analysis needs.

---

**Happy analyzing! 🚶‍♂️📊**