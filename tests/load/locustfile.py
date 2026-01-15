"""Locust load testing for Marketing Data Intelligence API."""

import random

from locust import HttpUser, between, task


class MarketingAPIUser(HttpUser):
    """Simulated user for load testing the API."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    # Sample data for requests
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"]
    questions = [
        "What are the best headphones?",
        "Show me laptops under $1000",
        "What are the top rated products?",
        "Find me good deals on electronics",
        "What's the best smartphone?",
        "Recommend kitchen appliances",
        "What are the highest rated books?",
        "Find running shoes with good reviews",
        "What products have the best discounts?",
        "Show me products with ratings above 4.5",
    ]

    def on_start(self):
        """Called when a simulated user starts."""
        # Check health on start
        self.client.get("/health")

    @task(3)  # Higher weight for predictions
    def predict_discount(self):
        """Test discount prediction endpoint."""
        payload = {
            "category": random.choice(self.categories),
            "actual_price": round(random.uniform(10, 5000), 2),
            "rating": round(random.uniform(1.0, 5.0), 1),
            "rating_count": random.randint(1, 10000),
        }
        
        with self.client.post(
            "/predict_discount",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                # Model not loaded - acceptable during testing
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)  # Medium weight for Q&A
    def answer_question(self):
        """Test Q&A endpoint."""
        payload = {
            "question": random.choice(self.questions),
        }
        
        with self.client.post(
            "/answer_question",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                # Service not available - acceptable during testing
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)
    def answer_question_with_filters(self):
        """Test Q&A endpoint with filters."""
        payload = {
            "question": random.choice(self.questions),
            "filter_category": random.choice(self.categories),
            "filter_max_price": random.choice([100, 500, 1000, 2000]),
            "filter_min_rating": random.choice([3.5, 4.0, 4.5]),
        }
        
        with self.client.post(
            "/answer_question",
            json=payload,
            catch_response=True,
        ) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)  # Lower weight for health checks
    def health_check(self):
        """Test health endpoint."""
        self.client.get("/health")

    @task(1)
    def get_model_status(self):
        """Test model status endpoint."""
        self.client.get("/predict/status")

    @task(1)
    def get_root(self):
        """Test root endpoint."""
        self.client.get("/")

    @task(1)
    def search_products(self):
        """Test product search endpoint."""
        query = random.choice(self.questions)
        params = {
            "query": query,
            "top_k": random.randint(3, 10),
        }
        
        with self.client.get(
            "/qa/search",
            params=params,
            catch_response=True,
        ) as response:
            if response.status_code in [200, 503]:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class HeavyLoadUser(HttpUser):
    """User that creates heavier load with burst requests."""

    wait_time = between(0.1, 0.5)  # Faster requests

    @task
    def burst_predictions(self):
        """Burst prediction requests."""
        for _ in range(5):
            payload = {
                "category": "Electronics",
                "actual_price": random.uniform(100, 1000),
                "rating": 4.5,
                "rating_count": 500,
            }
            self.client.post("/predict_discount", json=payload)


# Configuration for different load scenarios
# Run with: locust -f locustfile.py --host=http://localhost:8000

# Light load: locust -u 10 -r 2 --run-time 1m
# Medium load: locust -u 50 -r 10 --run-time 5m
# Heavy load: locust -u 200 -r 50 --run-time 10m
