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

# URL of the job listings page (changed to 'software developer')
url = "https://www.dice.com/jobs?q=software%20developer&location=New%20Jersey,%20USA&latitude=40.0583238&longitude=-74.4056612&countryCode=US&locationPrecision=State&adminDistrictCode=NJ&radius=30&radiusUnit=mi&page=1&pageSize=100&language=en"

# Open the URL
driver.get(url)

# Wait for job cards to load (using WebDriverWait)
try:
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, '#searchDisplay-div')))
    print("Page loaded successfully.")
except Exception as e:
    print(f"Error loading the page: {e}")
    driver.quit()

# Add a sleep to wait for the page to load completely
time.sleep(5)

# List to store job details
jobs = []

# Find all job card elements
try:
    # The common CSS selector for all job cards
    job_cards = driver.find_elements(By.CSS_SELECTOR, 'dhi-search-card')

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

            # Append job details to the list
            jobs.append({
                'Job Role': job_role,
                'Company': company_name,
                'Location': job_location,
                'Job Type': job_type
            })

        except Exception as inner_e:
            print(f"Error extracting data from a job card: {inner_e}")
            continue

except Exception as e:
    print(f"Error scraping the job data: {e}")

# If jobs were scraped, save them to a CSV file
if jobs:
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(jobs)
    df.to_csv('software_developer_job_listings.csv', index=False)
    print("All job listings have been successfully scraped and saved to 'software_developer_job_listings.csv'.")
else:
    print("No job listings were scraped.")

# Close the browser
driver.quit()
