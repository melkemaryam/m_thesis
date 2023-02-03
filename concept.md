# MSc Thesis Concept

Author: Hannah Claus

## Areas:

* NLP
    * Sentiment Analysis
    * Opinion Mining
* Social Media/News
    * Impact of negative/positive media
* Detection
    * of fake news
    * of "good"/"bad" news

## Background and Idea

The basic idea is to create a news platform which focuses on 'good' news from around the world. During the Covid-19 pandemic when the sole topic on the news was the virus and the corresponding death toll, the negative impact of these daily news on people's mental health was severe. And even after the worst part of the pandemic was over, there were so many other horrible things happening in the world, i. e. economic issues, ongoing wars, more epidemics and climate change. Of course, it is important to know what is going on, but regarding mental health or just general peace of mind, it would be a good asset to be able to filter for positive news from time to time.

Instead of going through various international news providers manually, a neural network could be trained to learn how to discriminate good from bad news. Of course, the terms good and bad would need to be defined first, hence, a new dataset will be created and annotated using a specific protocol that will make it possible for future additions. Using sentiment analysis, the classifier can be trained and tested on real news. While classifying good/bad news, the first step would be to include fake news detection as well, if the data includes social media posts.

Using this neural network, good news can be shown on a platform online. This can even be extended to having email subscriptions and regular blog posts, however, this has nothing to do with the neural network itself, it is merely an outlook.

Once the classifier is ready, further features can be added, such as:
* user validation
    * i.e. thumbs up/down to check whether the user thinks a certain piece of news has been correctly classified as positive
* personal news
    * "good"/"bad" news are subjective to each person
    * create a recommendation algorithm that personalises the news feed of each person
    * i.e. what one person in one country might deem as positive, another person in another country might not.
    * different ethnic backgrounds, beliefs, genders, societal and monetary aspects all influence the perception of "good"/"bad" news

## Annotations

* Like [Sentiment Annotated Dataset of Croatian News](https://www.clarin.si/repository/xmlui/handle/11356/1342)
    * using a five-level Likert scale (1—very negative, 2—negative, 3—neutral, 4—positive, and 5—very positive)

## Literature

### On sentiment analysis in social media/news:

* [Balahur, A., Steinberger, R., Kabadjov, M., Zavarella, V., Van Der Goot, E., Halkia, M., Pouliquen, B. and Belyaeva, J., 2013. Sentiment analysis in the news. arXiv preprint arXiv:1309.6202.](https://arxiv.org/abs/1309.6202)
* [O. Ajao, D. Bhowmik and S. Zargari, "Sentiment Aware Fake News Detection on Online Social Networks," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 2507-2511, doi: 10.1109/ICASSP.2019.8683170.](https://ieeexplore.ieee.org/abstract/document/8683170?casa_token=c8Mv3OGL4MYAAAAA:zFKfNSR_WmVm7fCHKW4qmgkAhtenQNlGOucAQ-VjFmy5o6FmY1N1NUN9_BZ4kVAx-q6hmBiWk8s)
* [B. Bhutani, N. Rastogi, P. Sehgal and A. Purwar, "Fake News Detection Using Sentiment Analysis," 2019 Twelfth International Conference on Contemporary Computing (IC3), 2019, pp. 1-5, doi: 10.1109/IC3.2019.8844880.](https://ieeexplore.ieee.org/abstract/document/8844880?casa_token=dTIveemm9j4AAAAA:RpZ-nGPe81N6GqPoSqP-NkRcOjD7aKn7DwAehEsIdjvyJk7ka34PSq7LHWAeBLqb3CNTNHuZ5oY)

* [Balahur, A. and Steinberger, R., 2009. Rethinking Sentiment Analysis in the News: from Theory to Practice and back. Proceeding of WOMSA, 9, pp.1-12.](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=533e1e815d569820d85d093e00e5b3261fd9332a)
* [Balahur, A., Steinberger, R., Van Der Goot, E., Pouliquen, B. and Kabadjov, M., 2009, September. Opinion mining on newspaper quotations. In 2009 IEEE/WIC/ACM International Joint Conference on Web Intelligence and Intelligent Agent Technology (Vol. 3, pp. 523-526). IEEE.](https://ieeexplore.ieee.org/abstract/document/5285040?casa_token=ye77JwbVdkEAAAAA:WTdPaZQ8LthSqbq7HAXEBMeqAVrvROxuJdY8eLHR_p1ZL6ZW304YF-qvvnZPEYBM5dPL86h9K-g)
* [Boldrini, E., Balahur, A., Martínez-Barco, P. and Montoyo, A., 2010, July. EmotiBlog: a finer-grained and more precise learning of subjectivity expression models. In Proceedings of the fourth linguistic annotation workshop (pp. 1-10).](https://aclanthology.org/W10-1801.pdf)
* [Padmaja, S., Fatima, S.S. and Bandu, S., 2013, July. Analysis of sentiment on newspaper quotations: A preliminary experiment. In 2013 Fourth International Conference on Computing, Communications and Networking Technologies (ICCCNT) (pp. 1-5). IEEE.](https://ieeexplore.ieee.org/abstract/document/6726650)
* [Balahur, A., Kabadjov, M., Steinberger, J., Steinberger, R. and Montoyo, A., 2009, December. Summarizing opinions in blog threads. In Proceedings of the 23rd Pacific Asia Conference on Language, Information and Computation, Volume 2 (pp. 606-613).](https://aclanthology.org/Y09-2019.pdf)
* [Padmaja, S., Fatima, S.S., Bandu, S., Kosala, P. and Abhignya, M.C., 2014, August. Comparing and evaluating the sentiment on newspaper articles: A preliminary experiment. In 2014 Science and Information Conference (pp. 789-792). IEEE.](https://ieeexplore.ieee.org/abstract/document/6918276?casa_token=xQnApsPDC_YAAAAA:adP6YwHnfikvN4zFJ3fRSijkblgm76JKoHKDIkcU2MPuukD2RDzNcLcfTQBVl1It6PbrOp-ROOw)
* [Padmaja, S., Fatima, S.S. and Bandu, S., 2014. Evaluating sentiment analysis methods and identifying scope of negation in newspaper articles. Int. J. Adv. Res. Artif. Intell, 3(11).](https://pdfs.semanticscholar.org/c5db/627ecd60e7e1226002ccfc99724c3e197dea.pdf)
* [Kaya, M., Fidan, G. and Toroslu, I.H., 2012, December. Sentiment analysis of Turkish political news. In 2012 IEEE/WIC/ACM International Conferences on Web Intelligence and Intelligent Agent Technology (Vol. 1, pp. 174-180). IEEE.](https://ieeexplore.ieee.org/abstract/document/6511881?casa_token=J2QA4wwy3x0AAAAA:vWcFo00_8w0LtcWbPUbWO1VGPQwSLbVZBfU9uNU6q1ULLCs-MHdBRxEkow56644not7HHI4XV44)