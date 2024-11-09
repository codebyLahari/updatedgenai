import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set up WebDriver using ChromeDriverManager
options = webdriver.ChromeOptions()
options.add_argument("--disable-gpu")  # Disable GPU rendering if it's causing issues

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# List of locations to search for software developer jobs
locations = [
    {"location": "New York, NY", "latitude": "40.712776", "longitude": "-74.005974"},
    {"location": "San Francisco, CA", "latitude": "37.774929", "longitude": "-122.419418"},
    {"location": "Austin, TX", "latitude": "30.267153", "longitude": "-97.743057"},
    {"location": "Seattle, WA", "latitude": "47.606209", "longitude": "-122.332069"}
]

# Base URL format for Dice
base_url = "https://www.dice.com/jobs?q=software%20developer&location={}&latitude={}&longitude={}&countryCode=US&locationPrecision=City&radius=30&radiusUnit=mi&page={}&pageSize=100&language=en"

# List to store all job details from different locations
all_jobs = []

# Maximum number of pages to scrape for each location
max_pages = 5

# Loop through each location in the list and scrape job listings
for loc in locations:
    previous_jobs = set()  # Store jobs from the previous page to detect duplicates
    for page in range(1, max_pages + 1):
        # Create the URL for the current location and page
        url = base_url.format(loc['location'].replace(" ", "%20"), loc['latitude'], loc['longitude'], page)

        # Open the URL for the current location and page
        driver.get(url)

        # Wait for job cards to load (using WebDriverWait)
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#searchDisplay-div')))
            print(f"Page {page} loaded successfully for {loc['location']}.")
        except Exception as e:
            print(f"Error loading the page {page} for {loc['location']}: {e}")
            break  # Stop trying further pages for this location if there's an issue

        # Add a sleep to wait for the page to load completely
        time.sleep(5)

        # Find all job card elements
        try:
            job_cards = driver.find_elements(By.CSS_SELECTOR, 'dhi-search-card')

            # If no job cards are found, break the loop (no more jobs on the next pages)
            if not job_cards:
                print(f"No more job listings found for {loc['location']} on page {page}.")
                break

            # Set to store jobs on the current page to detect if there are new jobs
            current_jobs = set()

            # Loop through each job card and scrape details
            for card in job_cards:
                try:
                    # Extract job role
                    job_role = card.find_element(By.CSS_SELECTOR, 'div.overflow-hidden > div > a').text

                    # Extract company name
                    company_name = card.find_element(By.CSS_SELECTOR, 'div.overflow-hidden > div > span').text

                    # Extract job location
                    job_location = card.find_element(By.CSS_SELECTOR, 'div.overflow-hidden > div > span').text

                    # Extract job type
                    job_type = card.find_element(By.CSS_SELECTOR, 'div.card-body.font-small.m-left-20.mobile-m-left-10 > div.d-flex.flex-wrap > div.card-position-type > span').text

                    # Create a unique identifier for the job (using job role and company name)
                    job_id = f"{job_role}-{company_name}"

                    # Add job to the set of current jobs for comparison
                    current_jobs.add(job_id)

                    # If the job is not a duplicate, add it to the final list
                    if job_id not in previous_jobs:
                        all_jobs.append({
                            'Job Role': job_role,
                            'Company': company_name,
                            'Location': job_location,
                            'Job Type': job_type,
                            'Search Location': loc['location'],  # Add the search location to the record
                            'Page': page  # Keep track of which page the job was found on
                        })

                except Exception as inner_e:
                    print(f"Error extracting data from a job card on page {page} in {loc['location']}: {inner_e}")
                    continue

            # If the current page's jobs are the same as the previous page, stop further scraping
            if current_jobs == previous_jobs:
                print(f"No new job listings found for {loc['location']} on page {page}. Stopping pagination.")
                break

            # Update the previous jobs to the current jobs for the next page comparison
            previous_jobs = current_jobs

        except Exception as e:
            print(f"Error scraping the job data on page {page} for {loc['location']}: {e}")
            break

# If jobs were scraped, save them to a CSV file
if all_jobs:
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(all_jobs)
    df.to_csv('software_developer_job_listings_with_pagination.csv', index=False)
    print("All job listings from multiple locations and pages have been successfully scraped and saved to 'software_developer_job_listings_with_pagination.csv'.")
else:
    print("No job listings were scraped.")

# Close the browser
driver.quit()
