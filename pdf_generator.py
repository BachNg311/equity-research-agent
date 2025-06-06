import markdown
import pdfkit
from pathlib import Path

# Merge content
files = [
    ("Macroeconomic Overview", "us_market_analysis.md"),
    ("Fundamental Analysis", "fundamental_analysis.md"),
    ("Technical Analysis", "technical_analysis.md"),
]

sections_html = ""
for title, filename in files:
    content = Path(filename).read_text(encoding='utf-8')
    html = markdown.markdown(content)
    sections_html += f"<h2>{title}</h2>\n{html}<hr>"

# Wrap in HTML template
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
        }}
        h1 {{
            text-align: center;
        }}
        h2 {{
            color: #000000;
        }}
    </style>
</head>
<body>
    <h1>US Equity Research Report</h1>
    {sections_html}
</body>
</html>
"""

# Write to file
Path("report.html").write_text(full_html, encoding='utf-8')

# Generate PDF
config = pdfkit.configuration(wkhtmltopdf="C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe")
pdfkit.from_file("report.html", "US_equity_research_report.pdf", configuration=config)
