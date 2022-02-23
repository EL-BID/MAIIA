import os
import scrapy

import gim_cv.config as cfg

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging

import logging


log = logging.getLogger(__name__)

# test this - original commented out
#configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})

settings_file_path = 'gim_cv.scrapers.vl_orthos.settings'
os.environ['SCRAPY_SETTINGS_MODULE'] = settings_file_path

scrapy_settings = get_project_settings()


def scrape_ortho_urls(settings=scrapy_settings):
    crawler_process = CrawlerProcess(settings)
    crawler_process.crawl('ortho_cataloguer')
    crawler_process.start() # the script will block here until the crawling is finished

"""
class Scraper:
    def __init__(self):
        settings_file_path = 'scraper.scraper.settings' # The path seen from root, ie. from main.py
        os.environ.setdefault('SCRAPY_SETTINGS_MODULE', settings_file_path)
        self.process = CrawlerProcess(get_project_settings())
        self.spider = QuotesSpider # The spider you want to crawl

    def run_spiders(self):
        self.process.crawl(self.spider)
        self.process.start()  # the script will block here until the crawling is finished
"""
