# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import MapCompose, TakeFirst
from w3lib.html import remove_tags

class InfomoneyNewsItem(scrapy.Item):
    # item field for news title
    title = scrapy.Field(
        input_processor=MapCompose(remove_tags)
    )
    
    # item field for news text
    text = scrapy.Field(
        input_processor=MapCompose(remove_tags)
    )
    pass
