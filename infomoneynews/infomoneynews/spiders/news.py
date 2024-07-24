from scrapy.spiders import SitemapSpider
from ..items import InfomoneynewsItem
from scrapy.loader import ItemLoader

def clean_article(article):
    clean = []

    for paragraph in article:
        if paragraph == 'Publicidade' or paragraph == 'Continua depois da publicidade' or '\n' in paragraph:
            clean.append('0')
        else:
            clean.append(paragraph)

    clean = [i for i in clean if i != '0']
    return clean

class NewsSpider(SitemapSpider):
    name = "news"
    allowed_domains = ["infomoney.com.br"]
    sitemap_urls = ["https://www.infomoney.com.br/news-sitemap.xml"]

    def parse(self, response):
        
        items = InfomoneynewsItem()

        title = response.css("title::text").extract()

        article = response.css("p::text").extract()
        clean = clean_article(article)

        items['title'] = title
        items['text'] = clean

        yield items
