import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from fpdf import FPDF
import os

# Base URL of the webpage you want to scrape
base_url = "http://sunshinefarmma.com/"

# File to store the failed attempts
failure_log_file = "pdf_creation_failures.txt"

# Make a request to get the HTML content of the page
response = requests.get(base_url)

# Check if the request was successful
if response.status_code == 200:
    # Set encoding explicitly to handle special characters
    response.encoding = response.apparent_encoding  # Set encoding to what BeautifulSoup detects

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all anchor tags with href attributes
    links = soup.find_all('a', href=True)

    # Create a directory to store the PDF files
    os.makedirs('pdfs', exist_ok=True)

    # Clear previous log file content
    with open(failure_log_file, "w") as log_file:
        log_file.write("Failed to create PDFs for the following URLs:\n\n")

    for link in links:
        # Convert relative links to absolute URLs
        full_url = urljoin(base_url, link['href'])

        # Only include sub-links from the same domain (sunshinefarmma.com)
        if base_url in full_url:
            try:
                # Make a request to the sub-link
                sub_response = requests.get(full_url)

                # Check if the request was successful
                if sub_response.status_code == 200:
                    # Set encoding explicitly for sub-pages as well
                    sub_response.encoding = sub_response.apparent_encoding

                    # Parse the sub-link content
                    sub_soup = BeautifulSoup(sub_response.content, 'html.parser')

                    # Extract all text from the sub-page
                    text = sub_soup.get_text(separator="\n", strip=True)

                    # Clean the text to remove problematic characters
                    cleaned_text = text.encode('ascii', 'ignore').decode('ascii')

                    # Create a PDF for the sub-page
                    pdf = FPDF()
                    pdf.add_page()

                    # Set font for the PDF
                    pdf.set_font("Arial", size=12)

                    # Add title
                    title = link.get_text().strip() or "No Title"
                    pdf.cell(200, 10, txt=title, ln=True, align="C")

                    # Add the extracted (cleaned) text to the PDF
                    for line in cleaned_text.split("\n"):
                        pdf.multi_cell(0, 10, line)

                    # Save the PDF with the link's text as filename
                    safe_title = title.replace("/", "_").replace("\\", "_").replace(" ", "_")
                    file_name = os.path.join('pdfs', f"{safe_title}.pdf")
                    pdf.output(file_name)
                    print(f"PDF created for: {full_url}")

                else:
                    # Log failed HTTP requests
                    with open(failure_log_file, "a") as log_file:
                        log_file.write(f"Failed to retrieve {full_url} (HTTP status code: {sub_response.status_code})\n")

            except Exception as e:
                # Log any other errors during the PDF creation process
                with open(failure_log_file, "a") as log_file:
                    log_file.write(f"Failed to create PDF for {full_url}: {e}\n")

    print(f"All available PDFs have been created. Check '{failure_log_file}' for any failures.")
else:
    print(f"Failed to retrieve the main page. Status code: {response.status_code}")
