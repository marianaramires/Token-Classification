from scrapy.spiders import SitemapSpider
from ..items import InfomoneyNewsItem
from scrapy.loader import ItemLoader

# removes 'trash' from text and joins the list
def clean_article(article):
    clean = []
    
    for line in article:
        if ("\n" not in line) and (not line.startswith("Baixe uma")) and (not line.startswith("Leia mais:")):
            clean.append(line)

    text = ' '.join(clean)
    return(text)

class NewsSpider(SitemapSpider):
    name = "news"
    allowed_domains = ["infomoney.com.br"]
    sitemap_urls = ["https://www.infomoney.com.br/post-sitemap481.xml",
                    "https://www.infomoney.com.br/post-sitemap480.xml",
                    "https://www.infomoney.com.br/post-sitemap479.xml",
                    "https://www.infomoney.com.br/post-sitemap478.xml",
                    "https://www.infomoney.com.br/post-sitemap477.xml",
                    "https://www.infomoney.com.br/post-sitemap476.xml"]

    def parse(self, response):
        
        items = InfomoneyNewsItem()

        title = response.css("title::text").extract()

        article = response.css("p:not(.py-2) ::text").extract()

        text = clean_article(article)

        items['title'] = title
        items['text'] = text

        yield items
