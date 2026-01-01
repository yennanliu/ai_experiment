#!/usr/bin/env python3
"""
Professional Resume PDF Generator

Converts markdown resume to professionally formatted PDF.
Optimized for ATS compatibility and clean business styling.
"""

import sys
import argparse
from pathlib import Path

def generate_with_weasyprint(input_file: str, output_file: str) -> bool:
    """Generate PDF using WeasyPrint (HTML/CSS method)"""
    try:
        from weasyprint import HTML, CSS
        from markdown import markdown

        # Read markdown content
        with open(input_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Convert markdown to HTML
        html_content = markdown(md_content, extensions=['extra', 'nl2br'])

        # Professional CSS styling
        css_style = """
        @page {
            size: letter;
            margin: 0.75in;
        }

        body {
            font-family: 'Calibri', 'Arial', 'Helvetica', sans-serif;
            font-size: 11pt;
            line-height: 1.15;
            color: #000000;
        }

        h1 {
            font-size: 22pt;
            font-weight: bold;
            margin-bottom: 8pt;
            text-align: center;
            letter-spacing: 0.5pt;
        }

        h2 {
            font-size: 13pt;
            font-weight: bold;
            text-transform: uppercase;
            margin-top: 14pt;
            margin-bottom: 8pt;
            border-bottom: 1.5pt solid #1a365d;
            padding-bottom: 2pt;
            letter-spacing: 1pt;
        }

        h3 {
            font-size: 11pt;
            font-weight: bold;
            margin-top: 8pt;
            margin-bottom: 4pt;
        }

        p {
            margin: 4pt 0;
        }

        ul {
            margin: 4pt 0;
            padding-left: 20pt;
        }

        li {
            margin: 3pt 0;
        }

        strong {
            font-weight: bold;
        }

        a {
            color: #1a365d;
            text-decoration: none;
        }

        /* Contact info styling */
        h1 + p {
            text-align: center;
            font-size: 10pt;
            margin-bottom: 12pt;
        }
        """

        # Generate PDF
        html = HTML(string=f"<html><body>{html_content}</body></html>")
        css = CSS(string=css_style)
        html.write_pdf(output_file, stylesheets=[css])

        return True

    except ImportError:
        print("❌ WeasyPrint not installed. Install with: pip install weasyprint markdown")
        return False
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False


def generate_with_pandoc(input_file: str, output_file: str) -> bool:
    """Generate PDF using Pandoc (LaTeX method)"""
    import subprocess

    try:
        cmd = [
            'pandoc',
            input_file,
            '-o', output_file,
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=0.75in',
            '-V', 'fontsize=11pt',
            '-V', 'mainfont=Calibri'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"❌ Pandoc error: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ Pandoc not installed. Install with: brew install pandoc (macOS) or apt-get install pandoc (Linux)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate professional PDF resume from markdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s resume.md                    # Generate resume.pdf
  %(prog)s resume.md -o MyResume.pdf    # Custom output name
  %(prog)s resume.md --method pandoc    # Use pandoc instead of weasyprint
        """
    )

    parser.add_argument('input', help='Input markdown file')
    parser.add_argument('-o', '--output', help='Output PDF file (default: input name with .pdf)')
    parser.add_argument(
        '--method',
        choices=['weasyprint', 'pandoc'],
        default='weasyprint',
        help='PDF generation method (default: weasyprint)'
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file '{args.input}' not found")
        return 1

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = str(input_path.with_suffix('.pdf'))

    print(f"📄 Generating PDF resume...")
    print(f"   Input:  {args.input}")
    print(f"   Output: {output_file}")
    print(f"   Method: {args.method}")
    print()

    # Generate PDF
    if args.method == 'weasyprint':
        success = generate_with_weasyprint(args.input, output_file)
    else:
        success = generate_with_pandoc(args.input, output_file)

    if success:
        print(f"✅ PDF generated successfully: {output_file}")
        print()
        print("📋 Next steps:")
        print("   1. Review the PDF for formatting and content accuracy")
        print("   2. Test with ATS checker (jobscan.co or resumeworded.com)")
        print("   3. Rename file to: FirstName_LastName_Resume.pdf")
        return 0
    else:
        print("❌ PDF generation failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
