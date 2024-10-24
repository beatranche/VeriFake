import os
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import concurrent.futures
import time

def scrape_page(page_number):
    """
    Scrapes data from a single page of the website.

    :param page_number: The number of the page to scrape.
    :return: A list of dictionaries with the scraped data.
    """
    driver = webdriver.Chrome()
    data = []  # Store the data for a single page
    cont = 0

    try:
        # Open the page
        driver.get(f'https://dbkf.ontotext.com/#!/searchViewResults?orderBy=score&lang=es&page={page_number}')

        # Wait until the h5 elements are present
        h5_elements = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'h5.claim-text'))
        )

        # Iterate over all h5 elements
        for index in range(len(h5_elements)):
            try:
                # Re-obtain the h5 elements to avoid stale element references
                h5_elements = WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'h5.claim-text'))
                )

                h5 = h5_elements[index]
                claim_text = h5.text
                
                # Find the <a> link inside the <h5> and click it
                link = h5.find_element(By.TAG_NAME, 'a')
                link.click()  # Click the link to open the additional content
                
                # Wait until the <p> element with class 'ng-binding' is present
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "quotes"))
                )

                # Get all <p> elements with class 'ng-binding'
                p_elements = driver.find_elements(By.CSS_SELECTOR, '.card-body p.ng-binding')
                links = driver.find_elements(By.CSS_SELECTOR, "ul.list-group a")
                
                result_links = []
                ignore_link = "https://archive"
                ignore_maldita = 'https://maldita.es/'

                # Iterate over all <p> elements
                for p in p_elements:
                    if p.text != '':
                        headline_text = p.text
                        data.append({'Index': cont, 'Headline': headline_text})
                    # Iterate over all <a> links
                    for link in links:
                        href = link.get_attribute('href')
                        if ignore_link not in href and ignore_maldita not in href:
                            result_links.append(href)
                            data.append({'Index': cont, 'Link': href})
                            
                            # Stop once you have 4 links
                            if len(result_links) == 4:
                                break
                
                cont += 1
                driver.back()  # Return to the previous page
                
            except StaleElementReferenceException:
                print("Stale element reference, retrying...")

    except Exception as e:
        print(f"Error on page {page_number}: {e}")
        driver.refresh()
        time.sleep(1)
        driver.back()
    
    driver.quit()
    return data  # Return the data from this page

def main(pages, workers):
    """
    Main entry point for the script. Scrapes web pages and saves the data to a JSON and CSV file.
    
    :param pages: Number of pages to scrape.
    :param workers: Number of workers for scraping.
    """
    pages = range(1, pages + 1)  # Iterate over pages 1 to pages
    
    all_data = []

    # Use ThreadPoolExecutor to manage threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(scrape_page, pages))

    # Combine all results into a single list
    for result in results:
        all_data.extend(result)
    
    return all_data  # Return combined data from all pages


 