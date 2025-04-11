import os
import re
from collections import Counter
import logging
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CVUpdater:
    def __init__(self):
        # Common English stop words that don't add meaning
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'when', 'where', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
            'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themselves', 'who', 'whom',
            'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'would', 'should', 'could', 'ought', 'in', 'on',
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'of', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
        
    def extract_keywords(self, text, top_n=20):
        """Extract most relevant keywords from text using simple word frequency"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        for char in string.punctuation:
            text = text.replace(char, ' ')
        
        # Split into words
        words = text.split()
        
        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Count word frequency
        word_counts = Counter(filtered_words)
        
        # Get top N keywords by frequency
        return [word for word, _ in word_counts.most_common(top_n)]
    
    def ai_extract_keywords(self, text, top_n=20):
        """
        This is a placeholder for using an AI API to extract keywords.
        In a real implementation, you would call an API like OpenAI, Google Cloud NLP, etc.
        
        For now, we'll use our simple implementation above.
        """
        logging.info("Using simple keyword extraction (in production, replace with AI API call)")
        return self.extract_keywords(text, top_n)
        
        # Example of how you might implement this with an AI API:
        """
        import requests
        
        api_key = "your_api_key"
        url = "https://api.openai.com/v1/completions"
        
        prompt = f"Extract the top {top_n} keywords from this job description that would be important for a resume:\n\n{text}\n\nKeywords:"
        
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-3.5-turbo-instruct",
                "prompt": prompt,
                "max_tokens": 100
            }
        )
        
        result = response.json()
        keywords_text = result['choices'][0]['text'].strip()
        keywords = [k.strip().lower() for k in keywords_text.split(',')]
        return keywords[:top_n]
        """
    
    def read_file(self, file_path):
        """Read a file and return its content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def write_file(self, file_path, content):
        """Write content to a file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            logging.info(f"Successfully wrote file: {file_path}")
        except Exception as e:
            logging.error(f"Error writing file {file_path}: {e}")
    
    def update_cv_sections(self, cv_content, keywords):
        """Update CV sections to emphasize keywords."""
        # Find experience section in CV
        sections = cv_content.split("--------------------------------------------------------------------------------")
        
        updated_sections = []
        for section in sections:
            # Modify the objective section to include keywords
            if "Objective" in section:
                # Extract the current objective text
                lines = section.strip().split('\n')
                heading = lines[0]
                if len(lines) > 1:
                    # Keep the heading, update the content with keywords
                    objective_text = "\n".join(lines[1:])
                    
                    # Ensure some important keywords are included in the objective
                    highlighted_keywords = [k for k in keywords[:8] if k not in objective_text.lower()]
                    if highlighted_keywords:
                        addition = f" Particularly skilled in {', '.join(highlighted_keywords)}."
                        objective_text += addition
                        
                    section_content = f"{heading}\n{objective_text}"
                    updated_sections.append(section_content)
                else:
                    updated_sections.append(section)
            
            # Update Technical Skills section to emphasize keywords
            elif "Technical Skills" in section:
                lines = section.strip().split('\n')
                heading = lines[0]
                skills_text = "\n".join(lines[1:])
                
                # Ensure important technical keywords are included
                tech_keywords = [k for k in keywords if k not in skills_text.lower() 
                                and k not in ['experience', 'team', 'work', 'years']]
                if tech_keywords:
                    # Add missing keywords to relevant skill categories
                    for keyword in tech_keywords[:5]:  # Limit to top 5 missing keywords
                        if not re.search(f"[^a-zA-Z]{keyword}[^a-zA-Z]", skills_text.lower()):
                            # Simple logic to add to appropriate skill category
                            if keyword in ['java', 'python', 'c++', 'javascript', 'go', 'node.js']:
                                skills_text = skills_text.replace('Languages:', f'Languages: {keyword.capitalize()},')
                            elif keyword in ['docker', 'kubernetes', 'git', 'jenkins']:
                                skills_text = skills_text.replace('Tools:', f'Tools: {keyword.capitalize()},')
                            elif keyword in ['aws', 'azure', 'cloud']:
                                skills_text = skills_text.replace('Cloud:', f'Cloud: {keyword.upper()},')
                            else:
                                skills_text = skills_text.replace('Other:', f'Other: {keyword.capitalize()},')
                
                updated_sections.append(f"{heading}\n{skills_text}")
            
            # Update experience section to emphasize keywords
            elif "Professional Experience" in section or "Experience" in section:
                # Keep the section as is for now, we could enhance this to modify bullet points
                updated_sections.append(section)
            
            else:
                updated_sections.append(section)
        
        # Reassemble the CV with proper spacing
        updated_cv = "--------------------------------------------------------------------------------".join(updated_sections)
        
        # Fix any formatting issues that might have occurred
        updated_cv = re.sub(r'--------------------------------------------------------------------------------([A-Za-z])', 
                          r'--------------------------------------------------------------------------------\n\1', updated_cv)
        
        return updated_cv
    
    def process_job_descriptions(self):
        """Process all job descriptions and create tailored CVs."""
        # Get source CV
        src_cv_path = os.path.join('src_cv', 'cv.txt')
        src_cv_content = self.read_file(src_cv_path)
        
        if not src_cv_content:
            logging.error("Could not read source CV")
            return
        
        # Ensure output directory exists
        output_dir = 'output_cv'
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each JD
        jd_dir = 'jds'
        for jd_file in os.listdir(jd_dir):
            if jd_file.endswith('.txt'):
                jd_name = os.path.splitext(jd_file)[0]
                jd_path = os.path.join(jd_dir, jd_file)
                jd_content = self.read_file(jd_path)
                
                if not jd_content:
                    logging.error(f"Could not read JD: {jd_path}")
                    continue
                
                # Extract keywords from JD - using our simplified method
                # (Replace with ai_extract_keywords when ready to use AI API)
                keywords = self.ai_extract_keywords(jd_content)
                logging.info(f"Extracted keywords from {jd_file}: {', '.join(keywords)}")
                
                # Update CV based on keywords
                updated_cv = self.update_cv_sections(src_cv_content, keywords)
                
                # Write updated CV
                output_path = os.path.join(output_dir, f"cv_{jd_name}.txt")
                self.write_file(output_path, updated_cv)

if __name__ == "__main__":
    logging.info("Starting CV updater...")
    updater = CVUpdater()
    updater.process_job_descriptions()
    logging.info("CV update process completed.") 