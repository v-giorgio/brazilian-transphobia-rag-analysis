1. Coleta de dados

2. Limpeza:

2.1. Filtragem manual de vídeos não relacionados ao tema ou semanticamente irrelevantes
2.2. Filtragem manual de comentários não relacionados ao tema
2.3. Limpeza automática do texto (normalização, remoção de ruídos textuais, caracteres repetidos e remoção automática de comentários vazios)

3. Classificação:

Além da classificação da atitude discursiva e da presença de conteúdo ofensivo, este trabalho incorpora uma dimensão adicional de análise relacionada ao cunho do comentário, permitindo identificar padrões recorrentes como invalidação identitária, discursos religiosos, políticos ou ameaçadores. Essa abordagem multidimensional possibilita uma compreensão mais rica e contextualizada dos discursos analisados, indo além de classificações binárias tradicionais.
Estrutura:

Dimensão 1: atitude discursiva (hateful, supportive, neutral)
Dimensão 2: ofensividade (binário)
Dimensão 3: cunho discursivo (religião, invalidação de identidade, política, assédio sexual, pseudociência)

{
  "attitude": "hateful",
  "offensive": true,
  "discursive_aspects": ["religião", "invalidação de identidade"]
}


{
  "attitude": "hateful",
  "offensive": true,
  "discursive_aspects": [
    "política",
    "invalidação de identidade"
  ]
}

Exemplo de dataset final:

video_title,comment,atitude_manual,ofensivo_manual,cunho_manual,atitude_prob_supportive,atitude_prob_neutral,atitude_prob_hateful,ofensivo_prob_non_offensive,ofensivo_prob_offensive,cunho_prob_religiao,cunho_prob_politica,cunho_prob_inval_genero,cunho_prob_assedio,cunho_prob_pseudociencia
"Debate sobre políticas de gênero","vai transformar a direita em uma esquerda colorida e burra","hateful","offensive","politica;invalidação_de_genero",0.03,0.12,0.85,0.18,0.82,0.02,0.91,0.87,0.04,0.11


O dataset final contém, além do texto do comentário e do título do vídeo, anotações manuais referentes às três dimensões analisadas (atitude discursiva, ofensividade e cunho discursivo). Adicionalmente, são armazenadas as distribuições de probabilidade geradas pelo classificador para cada dimensão, permitindo análises quantitativas baseadas em soft labels e a investigação da incerteza do modelo.

-- Anotação com label studio:

A anotação manual dos dados foi realizada utilizando a ferramenta open-source Label Studio, implantada em infraestrutura de nuvem na plataforma Amazon Web Services (AWS). O serviço foi executado em uma instância Amazon EC2 dedicada, conteinerizada via Docker, garantindo isolamento de dependências, portabilidade e reprodutibilidade do ambiente de anotação. A persistência dos dados foi assegurada por meio de volume de armazenamento Amazon EBS (Elastic Block Store) acoplado à instância, onde foram mantidos o banco de dados interno da aplicação (SQLite), as configurações dos projetos e os registros de anotação. O acesso à plataforma ocorreu via interface web autenticada, com contas individuais por anotador, permitindo rastreabilidade completa das contribuições e suporte a múltiplas anotações independentes por item. Os dados armazenados e trafegados na infraestrutura AWS permaneceram criptografados em repouso e em trânsito, com controle de acesso restrito ao pesquisador responsável por meio de políticas de identidade e permissões (IAM), não sendo disponibilizados publicamente nem acessados por terceiros não autorizados. Rotinas de backup periódico dos volumes de dados foram realizadas para mitigar riscos de perda de informação. Nenhum serviço terceirizado de anotação foi utilizado, mantendo os dados integralmente sob controle da infraestrutura do projeto e em conformidade com os requisitos éticos e de segurança da pesquisa.