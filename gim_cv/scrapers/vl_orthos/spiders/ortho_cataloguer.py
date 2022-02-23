# -*- coding: utf-8 -*-
import logging

import scrapy

import gim_cv.scrapers.vl_orthos as vl_orthos

from scrapy.loader import ItemLoader
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy_splash import SplashRequest

from gim_cv.scrapers.vl_orthos.items import VLOrthoItem

# css selectors for download button and collapsable folders
sel_download_button_css = "#downloadBtn"
sel_folder_css = (".modal-dialog .modal-dialog__content ul.filetree "
                  "li.collapsable ul span.folder")

# lua script for splash to get list of links
show_ortho_links_lua = ("""
function main(splash, args)
  splash.private_mode_enabled = false
  splash:set_user_agent("{user_agent}")
  url = args.url
  assert(splash:go(url))
  assert(splash:wait(1))
  download_button = assert(splash:select(
  	"{sel_download_button_css}"
  ))
  download_button:mouse_click()
  assert(splash:wait(1))
  -- click all folders
  local elems = splash:select_all("{sel_folder_css}")
  for i, elem in ipairs(elems) do
    elem:mouse_click()
    splash:wait(0.5)
  end
  splash:set_viewport_full()
  return splash:html()
end
""")


def parse_title(title, meta_keys):
    """
    Parses a title string on the orthophoto catalogue website.

    e.g. str is
    'Orthofotomozaïek, kleinschalig, zomeropnamen, kleur, 1979-1990,
    Vlaanderen'

    Parameters
    ----------
    title:
        The scraped title of an orthophoto page on the flemish catalogue site
    meta_keys:
        The keys for each piece of information included in the title
    """
    rtn_dict = {'dataset':title}
    logging.warn(title)
    try:
        rtn_dict.update(
            dict(zip(
                meta_keys,
                title.split(','))
            )
        )
    except Exception as e:
        logging.warning(f"Failed to parse fields from string: {title}")
    return rtn_dict


class OrthoCataloguerSpider(CrawlSpider):
    name = 'ortho_cataloguer'
    allowed_domains = ['download.vlaanderen.be']
    start_urls = ['https://download.vlaanderen.be/Catalogus']

    # spoof a regular browser
    user_agent = ('Mozilla/5.0 (X11; Linux x86_64; rv:74.0) '
                  'Gecko/20100101 Firefox/74.0')
    # string parse fields for page title (csv)
    title_meta_keys = ('type', 'scale', 'season','colour', 'period', 'region')
    # link grabbing rule
    rules = (
        # follow links to orthos - not grootschalig (requires login)
        Rule(
            LinkExtractor(
                restrict_xpaths=[
                    "//a[contains(@title, 'Ortho')"
                    " and not(contains(@title, 'groot'))"
                    " and contains(@title, 'kleur')"
                    "]"
                ]
            ),
            callback='expand_folders',
            follow=True,
            process_request='set_user_agent'
        ),
    )

    # counter-anti-scraping: ensure headers always imply normal browser usage
    def start_requests(self):
        for s in self.start_urls:
            yield scrapy.Request(url=s, headers={'User-Agent':self.user_agent})

    def set_user_agent(self, request):
        request.headers['User-Agent'] = self.user_agent
        return request

    def test_cb(self, response):
        yield response.url

    def expand_folders(self, response):
        """ parse the html of the individual ortho pages
        """
        # grab the title (name of dataset e.g. Ortho Vlaanderen 2012...)
        title = response.xpath(
            '//section[@class="region"]/div/div/div/h1/text()'
        ).get()
        # parse it to resolve individual fields
        title_meta = parse_title(title, meta_keys=self.title_meta_keys)
        # submit splash JS req to expand downloadable folders and extract html
        # feed this to extract_orthos, passing the dataset title
        yield SplashRequest(
            url=response.url,
            callback=self.extract_orthos,
            endpoint="execute",
            args={
                'lua_source':show_ortho_links_lua.format(
                    user_agent=self.user_agent,
                    sel_download_button_css=sel_download_button_css,
                    sel_folder_css=sel_folder_css
                ),
                'wait':2
            },
            meta = title_meta
        )

    def extract_orthos(self, response):
        # find all file classes in expanded download folders
        sel_file_links = response.xpath('//span[@class="file"]')
        # loop over selectorlist, getting ortho links, names and filesizes
        for file in sel_file_links:
            item = VLOrthoItem()
            loader = ItemLoader(item=item,
                                selector=file)
            loader.add_value('page_url', response.url)
            loader.add_xpath('suffix', './/a/text()')
            loader.add_xpath('filesize', './/child::node()[2]')
            loader.add_xpath('filename', './/a/text()')
            loader.add_xpath('download_url', './/a/@href')
            # tag on the metadata associated with the whole dataset: year etc
            loader.add_value('dataset', response.request.meta.get('dataset'))
            for k in self.title_meta_keys:
                if k in response.request.meta and k in item.fields.keys():
                    loader.add_value(k, response.request.meta[k])
            yield loader.load_item()
