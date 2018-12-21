# Classification de documents

## Présentation du dataset

### Origine
Lors d'un procès mené par le gourvernement américain, 14 Millions de documents ont été collectés et numérisés pour attaquer cinq grands groupes américains du tabac. 
Un échantollins alléatoire de 3482 documents a été classé par des opérateurs suivant le type de documents, entre eux : lettres, email, notes, publicité, etc.

### Classes des documents:

Ces 3482 documents proviennent de 10 classes :

1. Advertisement
2. Email
3. Form
4. Letter
5. Memo
6. News
7. Note
8. Report
9. Resume
10. Scientific

Les classes sont reparties de la suivante façon : 

![Alt text](hist.png?raw=true)

## Problématique

Le but de ces travaux est d'entraîner un modèle pour qu'il soit capable de classifier automatiquement les documents. On rappelle que seulement 3482 documents sont classifiés pour l'instant.

À notre disposition, il y a :
1. Les images des documents
2. Le texte extrait de ces documents par un OCR
3. Les classes correspondantes à chaque document étiquettés par des opérateurs

Dans le cadre de ce cours, analyse de texte, nous allons traiter le texte extrait automatiquement de ces documents pour ainsi caractériser les documents et pouvoir les diviser par les catégories prédefinies.

Nous sommes donc confrontés à un problème de classification . Pour celà, nous avons choisi d'utiliser un modèle adapté à cette tâche, les réseaux de neurones convolutifs. Ceci est une méthode d'actualité très éfficace. Pour classifier les documents, nous n'avons pas besoin d'un composante temporelle car l'ordre des mots dans le documents n'est pas importe, mais plutôt leur sens . Pour celà, nous utilisons un CNN au lieu d'un LSTM, mais aussi, nous avons une couche d'embedding, avant la couche convolutif pour mieux répresenter les mots de chaque document sans s'abstenir de tout leur sens. 

Le texte, comme nous pouvons observer sur n'importe quel exemple, proviennent d'un OCR et ceci comporte beaucoup d'erreur, du à la qualité des documents scannés. Ceci veut dire que nos données sont très bruitées, donc, nous n'avons pas besoin de rajouter du bruit pour éviter le sur apprentissage. Par contre, encore une fois pour éviter le sur apprentissage, nous allons ajouter une couche de dropout, qui va nous obliger à s'emdébarasser d'une partie du réseau entraîné à chaque epoch.


## Expérimentations 

Beaucoup d'expérimentations ont été menées pour essayer de traiter au mieux ce problème:

⋅⋅* Ajout de couches
⋅⋅* Ajout de cellulles 
⋅⋅* Différentes tailles de séquences
⋅⋅* Différentes tailles de embedding
⋅⋅* Différentes taille de batch
⋅⋅* Maximum de caractéristique : c'est à dire le nombre de mots retenus
⋅⋅* Le dropout

Les paramètres retenus, ont été ceux qui nous ont permi d'obtenir une meilleure performance sur la base de developpement . Pour prévoir au mieux la performance en géneralization, une base de test à été mise de cotê jusqu'à la fin, sans qu'elle intervienne à aucun moment pour le choix du modèle.

D'après les expériences réalisées, l'idée retenue est que plus simples est le réseau, meilleur on est à apprendre les données et à predire. Donc l'ajout de couche et de neurones n'a pas été retenu, ainsi que des plus grosses tailles de vecteur de caractéristique ou de embedding.
Les données d'apprentissage sont bien séparées, très rapidement. En test, nous avons surement pas les mêmes performances (jusqu'à 99% en apprentissage et jusqu'à 72% pour le dev) . Malgrè les efforts pour éviter le surapprentissage, ce sont les meilleures performances que nous avons pu constater sur la base de dev.

Utiliser le script :

final.py : fichier qui permet d'entraîner le réseau et retient l'epoch qui obtient la meilleure performance sur la base de dev, pour tester sur la base de test.

arguments :
⋅⋅* -csv : chemin pour le fichier csv où sont sauvegardé le noms des imges et les labels associés.
⋅⋅* -dir_text : dossier où se trouve les textes 

example : 

python3 final.py -csv /home/avelina/Documentos/NLP/final/data/Tobacco3482.csv -dir_text /home/avelina/Documentos/NLP/final/data/Tabacco3482/


## Améliorations :

Une des expériences pas menée est de essayer de diminuer un peu des donnés de test, pour avoir plus d'examples en apprentissage . Ceci nous permet d'apprendre plus d'examples mais nous n'aurons pas une vrai idée de ses performances en géneralization.

Nous pouvons aussi, surement, construire un multicassifieur, pour rajouter les informations que les images peuvent nous donner (ceci ne s'inscrit pas vraiment dans le cadre de ce cours, c'est la raison pour laquelle n'a pas été mis en place). Un autre réseau, de convolution par exemple certainement plus profond pour essayer d'extraires des caractéristiques à tous les niveaux classifier les documents visuellement. Cette information, ajouté à celle que le texte nous apporte, peut améliorer significativement les performances.






