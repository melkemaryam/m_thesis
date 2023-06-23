# Sentiment analysis performed by chat-gpt

## Trial 1

### Prompt: 
What is the sentiment of the following news article, which is delimited with triple backticks? Give your answer as a single number, 2 for "positive", 1 for "neutral" or 0 for "negative". Give a probability score as well! Elaborate with one short and concise sentence on the next line. Format your response as a JSON object with "Sentiment", "Probability", "Explanation" as the keys. Article: ''' {article}''', Explain how you calculate the probability and sentiment values in a concluding sentence.

Article: [Parent Goes Viral On TikTok for Life-Saving Invention](https://www.goodgoodgood.co/articles/cpr-wrap)

Response: 

{ "Sentiment": 2, "Probability": 0.9, "Explanation": "The sentiment of the article is positive, with a high probability, as it highlights the importance of CPR Wrap in empowering people to save lives and has received widespread support and endorsement from viewers, including professionals in the field." }

To calculate the sentiment and probability values, I analyzed the overall tone of the article, which consistently emphasized the positive aspects of CPR Wrap and its potential to make a difference in emergency situations. The inclusion of stories about everyday heroes and the positive feedback from commenters further contributed to the positive sentiment. The high probability score reflects the strong evidence supporting a positive sentiment throughout the article.

## Trial 2

### Prompt: 
What is the sentiment of the following news article, which is delimited with triple backticks? Give your answer as a single number, 2 for "positive", 1 for "neutral" or 0 for "negative". Give a probability score as well! Elaborate with one short and concise sentence on the next line. Format your response as a JSON object with "Sentiment", "Probability", "Explanation" as the keys. After that, in a normal text format, explain how you calculate the probability and sentiment values in a concluding sentence. Additionally, generate a catchy but matching headline for the news article that corresponds to the article's sentiment. Article: '''{article}'''

Response: 

{ "Sentiment": 2, "Probability": 0.95, "Explanation": "The sentiment of the article is positive because it highlights the importance of CPR in emergency situations and praises the CPR Wrap tool for making CPR more accessible and easy to perform." }

The probability of the sentiment being positive is 0.95, indicating a high likelihood. The article emphasizes the significance of CPR in crisis situations and introduces the CPR Wrap tool as a simple and accessible solution for performing CPR effectively. It includes positive testimonials from viewers who support the invention, reinforcing the positive sentiment.

Catchy Headline: "Revolutionary CPR Wrap Empowers Everyday Heroes to Save Lives with Ease"