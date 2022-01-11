# Projet de Outils de Programmation de la science des données et intelligence des affaires

_Par Mathys Clerget (CLEM14050107)_

## Description
Ce projet est effectué dans le cadre du cours 8PRO408 à l’UQAC.
Il consistait à effectuer le travail sur un dataset composé de `Time Series`, appliquer notre analyse à un arbre de décision qui nous est fourni.
La présentation est visible via ce lien : [Panopto](https://uqac.ca.panopto.com/Panopto/Pages/Viewer.aspx?id=c787a397-8260-4873-91a0-ae03017cc282)

## Disposition du projet

Le projet est disposé en **4 parties** 
1. Files
2. Test
3. Train
4. Utils

### Dossier Files

Le dossier Files est composé de plusieurs sous-dossiers comportant les résultats produits lors de l’execution du projet :

1. File > Out > Pickles > 
    * Les fichiers pickles créés tout au long du projet
2. File > Out > PicklesPlots >
    * Les plots sauvegardés au format png des réductions de dimension de chaque fenêtre temporelle.
     * Les noms de chaques png sont tel cet exemple :
        * **DimReduc100-5.png**
            * `DimReduc`: un simple nom;
            * `100-` : la fréquence d’échantillonnage;
            * `5` : Le nombre de valeur de chevauchement.
3.  File > Out > Stats >
    * Les plots sauvegardés au format png des valeurs de chaque caractéristiques
        * Les noms de chaques png sont tel cet exemple :
        * **1_Standing still (1 min) _ACC_Chest_X.png**
            * `1_` : correspond à la classe;
            * `Standing still (1 min) ` : est l’activité;
            * `_ACC_Chest_X` : la caractéristique cible.

### Dossiers Test et Train

Ces dossiers possèdent les codes a éxécuter respectivement pour Le TestSet et le TrainSet

### Utils

Ce dossier possède plusieurs fichiers utiles à la bonne execution du projet : 

* Il est composé de plusieurs sous dossiers : 
    1. `DimUtils` : La réduction de dimension.
    2. `ExtractionUtils` : L’extraction des statistiques des caractéristiques.
    3. `TreeClassUtils` : Le code fourni par le professeur. 
* Mais aussi d’autres fichiers `python` :
    1. `Get.py` : composé de plusieurs getter.
    2. `Graphics.py` : composé d’une fonction de plotting
    3. `Protocol.py` : composé du protocol executé par `Train.py` et `Test.py`
    4. `Utils.py` : composé de fonctions diverses.

