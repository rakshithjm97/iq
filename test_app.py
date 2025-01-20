import unittest
from app import get_ai_response

class TestApp(unittest.TestCase):
    def setUp(self):
        self.kernel = MockKernel()

    def test_get_ai_response_valid_question(self):
        question = "What is the capital of France?"
        expected_response = "The capital of France is Paris."
        self.kernel.set_response(expected_response)
        response = get_ai_response(question, self.kernel)
        self.assertEqual(response, expected_response)

    def test_get_ai_response_no_response(self):
        question = "What is the meaning of life?"
        expected_response = "Sorry, I don't have an answer for that."
        self.kernel.set_response("")
        response = get_ai_response(question, self.kernel)
        self.assertEqual(response, expected_response)

    def test_get_ai_response_exception(self):
        question = "This will cause an exception"
        self.kernel.set_response(Exception("Test exception"))
        response = get_ai_response(question, self.kernel)
        self.assertTrue("Test exception" in response)

class MockKernel:
    def __init__(self):
        self.response = ""

    def set_response(self, response):
        self.response = response

    def respond(self, question):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

if __name__ == '__main__':
    unittest.main()