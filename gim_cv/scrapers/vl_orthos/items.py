# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
import logging
import os
import re
import timbermafia as tm
import scrapy


from itemloaders.processors import TakeFirst, MapCompose
#from scrapy.loader.processors import TakeFirst, MapCompose

log = logging.getLogger(__name__)


def extract_size(value):
    match = re.search("\d+\s?(k|M|G)B", value[0])
    if match is not None:
        return match[0]
    logging.warn(match)
    return 'not_found'


def infer_resolution(value):
    if 'kleinschalig' in value:
        return 1.
    elif 'middenschalig' in value:
        return .25
    elif 'grootschalig' in value:
        return 0.1
    else:
        return None


def extract_season(value):
    return value.strip(' opnamen ')


def extract_suffix(value):
    return value.split('.')[0].split('_')[-1]



class VLOrthoItem(scrapy.Item, tm.Logged):
    # define the fields for your item here like:
    dataset = scrapy.Field(output_processor=TakeFirst())
    filesize = scrapy.Field(input_processor=extract_size,#MapCompose(extract_size),
                            output_processor=TakeFirst()
                            )
    filename = scrapy.Field(output_processor=TakeFirst())
    download_url = scrapy.Field(output_processor=TakeFirst())
    page_url = scrapy.Field(output_processor=TakeFirst())
    type = scrapy.Field(output_processor=TakeFirst())
    scale  = scrapy.Field(output_processor=TakeFirst(),
                          input_processor=MapCompose(infer_resolution))
    season = scrapy.Field(output_processor=TakeFirst(),
                          input_processor=MapCompose(extract_season))
    period = scrapy.Field(output_processor=TakeFirst(),
                          input_processor=MapCompose((lambda x : x.strip())))
    region = scrapy.Field(output_processor=TakeFirst(),
                          input_processor=MapCompose((lambda x : x.strip())))
    suffix = scrapy.Field(output_processor=TakeFirst(),
                          input_processor=MapCompose(extract_suffix))
