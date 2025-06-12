import markdown
import pdfkit
from pathlib import Path

# Load the report.md file
markdown_path = Path("report.md")
markdown_text = markdown_path.read_text(encoding="utf-8")

# Convert Markdown to HTML (including tables, images, bold, etc.)
html_body = markdown.markdown(
    markdown_text,
    extensions=["fenced_code", "tables", "toc", "md_in_html"]
)

# Styled HTML wrapper with enlarged fonts
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Equity Research Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            color: #333;
            font-size: 18px;
            line-height: 1.8;
        }}
        h1 {{
            text-align: center;
            font-size: 40px;
            margin-bottom: 30px;
        }}
        h2 {{
            font-size: 28px;
            color: #2c3e50;
            border-bottom: 2px solid #ccc;
            padding-bottom: 8px;
            margin-top: 50px;
        }}
        h3 {{
            font-size: 22px;
            color: #34495e;
            margin-top: 30px;
        }}
        p, li {{
            font-size: 18px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 10px 14px;
            text-align: left;
            font-size: 16px;
        }}
        th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 30px auto;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 4px;
            font-size: 15px;
            font-family: Consolas, monospace;
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>
"""

# Save to intermediate HTML
Path("report_from_md.html").write_text(full_html, encoding="utf-8")

# Convert to PDF using wkhtmltopdf
config = pdfkit.configuration(wkhtmltopdf="C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe")
pdfkit.from_file(
    "report_from_md.html",
    "US_equity_report.pdf",
    configuration=config,
    options={
        "enable-local-file-access": "",
        "page-size": "Letter",
        "margin-top": "0.75in",
        "margin-right": "0.75in",
        "margin-bottom": "0.75in",
        "margin-left": "0.75in",
        "encoding": "UTF-8",
    }
)
