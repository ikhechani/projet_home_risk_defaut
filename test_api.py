import unittest
from fastapi.testclient import TestClient
from api import app  # Assurez-vous que le fichier de l'API s'appelle bien `api.py`

class TestAPI(unittest.TestCase):
    def setUp(self):
        # Créer un client de test pour l'API
        self.client = TestClient(app)

    def test_welcome_message(self):
        # Tester l'endpoint racine
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "Bienvenue")

    def test_check_client_id_valid(self):
        # Tester l'endpoint de vérification d'un client valide
        response = self.client.get("/302160")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json())

    def test_check_client_id_invalid(self):
        # Tester l'endpoint de vérification d'un client invalide
        response = self.client.get("/999999")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json())

    def test_get_prediction(self):
        # Tester l'endpoint de prédiction pour un client spécifique
        response = self.client.get("/prediction/302160")
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.json(), 0.4647581264389726, places=6)

if __name__ == "__main__":
    unittest.main()
