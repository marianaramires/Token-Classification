# Processo Seletivo do Laboratório de Inteligência Artificial
 Segunda fase do Processo Seletivo do Laboratório de Inteligência Artificial (LIA) da UFMS

## Problema: Token classification

Sua tarefa será extrair de textos coletados do infomoney as entidades de:   empresa,  empresario,  politico,  outras_pessoas, valor_financeiro, cidade,  estado,  pais,  organização e  banco.

Parte 1 - coleta de dados:
  Façam scrapy do site https://www.infomoney.com.br/
  sugestão: vejam que o site possui um robots.txt  

Parte 2 - rotulação:
   instale localmente em sua máquina label studio (https://labelstud.io/guide/install.html) e rotule as classes definidas no problema. Na interface do label studio vejam o template de Natural Language Processing -> Named Entity Recognition

Parte 3 - treinamento: treinar o modelo  

Parte 4 - avaliação: precision, recall, f1  

Parte 5 - deploy: colocar no label studio o modelo para fazer Pre-annotation

