# Implémentez un modèle de scoring

En tant que Data Scientist au sein de l’entreprise financière "Prêt à dépenser", spécialisée dans les crédits à la consommation pour les individus avec peu ou pas d’historique d'emprunt, votre mission est de contribuer à l’implémentation d’un modèle de scoring.

L’objectif de l’entreprise est de mettre en place un système de "scoring crédit", capable de calculer la probabilité qu’un client rembourse son prêt, puis de classer la demande en crédit accepté ou rejeté. Pour y parvenir, l'entreprise souhaite développer un algorithme de classification en s’appuyant sur une variété de données, telles que les informations comportementales et les données provenant d’autres institutions financières.

Un autre besoin a été soulevé par les responsables de la relation client : la demande croissante des clients pour plus de transparence sur les décisions de l’octroi de crédits. Cette exigence de transparence est en adéquation avec les valeurs que l’entreprise souhaite promouvoir. Ainsi, "Prêt à dépenser" a décidé de développer un dashboard interactif. Ce dernier permettra aux gestionnaires de la relation client non seulement d'expliquer les décisions d’octroi de crédit de manière claire et transparente, mais aussi d’offrir aux clients un accès facile à leurs informations personnelles pour une exploration approfondie.

Les données nécessaires au projet sont disponibles à l’adresse suivante : https://www.kaggle.com/c/home-credit-default-risk/data


## Objectif 
1.Développer un modèle de scoring qui prédit automatiquement la probabilité de défaut de paiement d'un client.
1. Créer un dashboard interactif destiné aux gestionnaires de la relation client, permettant d’interpréter les prédictions du modèle et d’améliorer la compréhension des clients par ces derniers.
3.Mettre en production le modèle de scoring via une API et déployer le dashboard interactif qui interagit avec cette API pour effectuer les prédictions.

## Organisation
Le projet est structuré en deux branches principales pour permettre un déploiement simplifié de l’API et du dashboard sur des URLs distinctes tout en minimisant les coûts. Vous trouverez dans la branche main les fichiers relatifs au déploiement de l’API, et dans la branche dashboard, ceux liés au déploiement du dashboard.

## Le dashboard
Le dashboard a été développé en utilisant Streamlit pour l’interface utilisateur et FastAPI pour l’API. Il a été déployé en premier lieu sur un environnement en local suite a quelques blocage .
