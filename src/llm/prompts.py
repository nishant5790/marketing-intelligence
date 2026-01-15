"""Prompt templates for the LLM assistant."""

# System prompt for the marketing assistant
SYSTEM_PROMPT = """You are an intelligent marketing assistant for an e-commerce platform. Your role is to help users find products, answer questions about our catalog, and provide recommendations.

Guidelines:
1. Always base your answers on the provided product context when available.
2. Be helpful, accurate, and concise in your responses.
3. If you don't have enough information to answer a question, say so clearly.
4. When discussing products, include relevant details like price, rating, and discount when available.
5. Never make up product information that isn't in the provided context.
6. Format prices in a user-friendly way (e.g., $99.99).
7. When comparing products, highlight key differences and similarities.
8. Provide actionable recommendations when appropriate.

Response Format:
- Be conversational but professional
- Use bullet points for lists when appropriate
- Highlight important information like deals or highly-rated products
"""

# Template for answering product-related questions
QA_PROMPT_TEMPLATE = """Based on the following product information from our catalog, please answer the user's question.

{context}

User Question: {question}

Instructions:
- Answer based ONLY on the information provided above
- If the information doesn't fully answer the question, acknowledge what you can answer and note any limitations
- Include specific product details (name, price, rating) when relevant
- Be helpful and suggest alternatives if the exact request can't be met

Answer:"""

# Template for product recommendations
RECOMMENDATION_PROMPT_TEMPLATE = """Based on the following products in our catalog, provide personalized recommendations.

{context}

User Request: {request}

Instructions:
- Recommend the most relevant products from the list above
- Explain why each recommendation is suitable
- Include key details: name, price, discount, and rating
- Prioritize best value (high rating + good discount) when relevant
- Suggest alternatives if the exact match isn't available

Recommendations:"""

# Template for product comparison
COMPARISON_PROMPT_TEMPLATE = """Compare the following products based on the user's criteria.

{context}

User Request: {request}

Instructions:
- Create a clear comparison highlighting key differences
- Consider: price, features, ratings, and value
- Provide a recommendation based on the comparison
- Note any missing information that could affect the comparison

Comparison:"""

# Template for discount/pricing questions
PRICING_PROMPT_TEMPLATE = """Based on the following product pricing information, answer the user's question.

{context}

User Question: {question}

Instructions:
- Focus on pricing, discounts, and value
- Calculate savings where relevant
- Mention if products have good value (high discount + good rating)
- Be transparent about price ranges

Answer:"""

# Few-shot examples for domain adaptation
FEW_SHOT_EXAMPLES = [
    {
        "question": "What are the best headphones under $100?",
        "context": """1. Sony WH-CH520
   Category: Electronics
   Price: $89.99
   Discount: 25%
   Rating: 4.5/5 (1,234 reviews)

2. JBL Tune 510BT
   Category: Electronics
   Price: $49.99
   Discount: 15%
   Rating: 4.3/5 (856 reviews)""",
        "answer": """Based on our catalog, here are the best headphones under $100:

**Top Pick: Sony WH-CH520** ($89.99)
- Rating: 4.5/5 from 1,234 reviews
- Currently 25% off
- Great balance of quality and price

**Budget Choice: JBL Tune 510BT** ($49.99)
- Rating: 4.3/5 from 856 reviews
- Currently 15% off
- Excellent value for the price

Both are highly rated by customers. The Sony offers slightly better quality, while the JBL is a great budget-friendly option.""",
    },
    {
        "question": "Are there any deals on kitchen appliances?",
        "context": """1. Instant Pot Duo 7-in-1
   Category: Home & Kitchen
   Price: $89.00
   Discount: 35%
   Rating: 4.7/5 (45,678 reviews)

2. Ninja Air Fryer
   Category: Home & Kitchen
   Price: $79.99
   Discount: 20%
   Rating: 4.6/5 (12,345 reviews)""",
        "answer": """Yes! We have some great deals on kitchen appliances right now:

**Best Deal: Instant Pot Duo 7-in-1** ($89.00)
- **35% OFF** - Great savings!
- Rating: 4.7/5 (45,678 reviews) - our highest rated kitchen appliance
- Versatile 7-in-1 functionality

**Also on Sale: Ninja Air Fryer** ($79.99)
- 20% off
- Rating: 4.6/5 (12,345 reviews)
- Perfect for healthy cooking

The Instant Pot is particularly popular with over 45,000 positive reviews. Both items are excellent value at their current sale prices.""",
    },
]


def build_qa_prompt(context: str, question: str) -> str:
    """Build a Q&A prompt with context.
    
    Args:
        context: Product context from retrieval.
        question: User's question.
        
    Returns:
        Formatted prompt string.
    """
    return QA_PROMPT_TEMPLATE.format(context=context, question=question)


def build_prompt_with_examples(context: str, question: str) -> str:
    """Build a prompt with few-shot examples.
    
    Args:
        context: Product context from retrieval.
        question: User's question.
        
    Returns:
        Formatted prompt with examples.
    """
    examples_text = "\n\n---\n\n".join([
        f"Example Question: {ex['question']}\n\nContext:\n{ex['context']}\n\nExample Answer:\n{ex['answer']}"
        for ex in FEW_SHOT_EXAMPLES[:2]  # Use first 2 examples
    ])
    
    return f"""Here are some examples of how to answer product questions:

{examples_text}

---

Now, please answer this question:

{QA_PROMPT_TEMPLATE.format(context=context, question=question)}"""
