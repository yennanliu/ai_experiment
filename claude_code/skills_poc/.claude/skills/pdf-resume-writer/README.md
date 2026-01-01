# PDF Resume Writer Skill

Professional resume/CV writer for Software Engineering positions at top tech companies (FAANG+).

## Features

- ✅ ATS-friendly formatting
- ✅ FAANG-optimized content structure
- ✅ Quantifiable achievement bullets
- ✅ Professional business styling
- ✅ PDF generation from markdown
- ✅ Templates for all experience levels

## Usage

### Using the Skill

```bash
# Invoke the skill
/pdf-resume-writer

# Or use directly
claude "Create a professional PDF resume for a senior software engineer"
```

### PDF Generation

#### Method 1: WeasyPrint (Recommended)

```bash
# Install dependencies
pip install weasyprint markdown

# Generate PDF
python generate_pdf.py resume.md -o John_Doe_Resume.pdf
```

#### Method 2: Pandoc

```bash
# Install pandoc
brew install pandoc  # macOS
# or
sudo apt-get install pandoc  # Linux

# Generate PDF
python generate_pdf.py resume.md --method pandoc
```

#### Method 3: Direct Commands

```bash
# Using pandoc directly
pandoc resume.md -o resume.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=0.75in \
  -V fontsize=11pt \
  -V mainfont="Calibri"

# Using weasyprint directly
weasyprint resume.html resume.pdf
```

## What This Skill Does

1. **Content Optimization**
   - Rewrites bullets using: Action Verb + Task + Technology + Metrics
   - Adds quantifiable achievements
   - Removes fluff and irrelevant content
   - Tailors content to FAANG requirements

2. **Professional Formatting**
   - Clean, ATS-friendly layout
   - Professional typography
   - Clear visual hierarchy
   - Standard section organization

3. **PDF Generation**
   - Converts markdown to professional PDF
   - Maintains ATS compatibility
   - Professional business styling
   - Proper margins and spacing

## Templates Included

- **Entry-Level**: New grads, 0-2 years experience
- **Mid-Level**: 3-7 years experience
- **Senior**: 8+ years experience, Staff+ engineers

## Best Practices

### Content
- Start bullets with action verbs
- Include quantifiable metrics
- Show scale and impact
- Highlight relevant technologies
- Keep it concise (1-2 pages)

### Format
- Use standard fonts (Calibri, Arial, Helvetica)
- 10-11pt body text
- 0.75in margins
- No tables, graphics, or text boxes
- PDF output only

## Example Usage

```python
# 1. Analyze existing resume
"Analyze my current resume for FAANG applications"

# 2. Optimize content
"Rewrite this work experience with quantifiable achievements"

# 3. Generate PDF
"Create a professional PDF resume from this markdown"
```

## File Structure

```
.claude/skills/pdf-resume-writer/
├── SKILL.md              # Main skill instructions
├── README.md             # This file
└── generate_pdf.py       # PDF generation script
```

## Requirements

**For PDF Generation:**
- Python 3.7+
- WeasyPrint: `pip install weasyprint markdown`
- OR Pandoc: `brew install pandoc` (macOS) / `apt-get install pandoc` (Linux)

**For Resume Review:**
- No additional requirements

## Tips

1. **ATS Testing**: Test generated PDFs at jobscan.co or resumeworded.com
2. **Tailoring**: Customize resume for each job application
3. **Keywords**: Match job description keywords naturally
4. **Metrics**: Always include numbers and percentages
5. **Proofreading**: Check for typos and consistency

## Common Pitfalls to Avoid

❌ Using fancy graphics or templates
❌ Listing responsibilities instead of achievements
❌ No quantifiable metrics
❌ Inconsistent formatting
❌ More than 2 pages
❌ Typos or grammatical errors

## Support

For issues or improvements:
- Check SKILL.md for detailed guidelines
- Review template examples
- Test PDF generation locally first

## License

Project skill - internal use
