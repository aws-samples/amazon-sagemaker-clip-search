## Use Amazon SageMaker and Amazon OpenSearch Service to implement unified text and image search with a CLIP model 


This repository aims at building a machine learning (ML) powered search engine prototype to retrieve and recommend products based on text or image queries. This is a step-by-step guide on how to create SageMaker Models with [Contrastive Language-Image Pre-Training (CLIP)](https://openai.com/blog/clip/), use the models to encode images and text into embeddings, ingest embeddings into [Amazon OpenSearch Service](https://aws.amazon.com/opensearch-service/) index, and query the index using OpenSearch Service [k-nearest neighbors (KNN) functionality](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html).


### <ins> Background </ins>

Embedding-based retrieval(EBR) is well used in search and recommendation systems. It uses nearest (approximate) neighbour search algorithms to find similar or closely related items from an embedding store (also known as a vector database). Classic search mechanisms depend heavily on key-word matching and ignore the lexical meaning or query’s context. The goal of EBR is to provide users with the ability to find the most relevant products using free text. It is popular because compared with key-word matching it leverages semantic concepts in retrieval process. 

In this repo, we focus on building a machine learning (ML) powered search engine prototype to retrieve and recommend products based on text or image queries. This uses the Amazon OpenSearch Service and its k-nearest neighbors (KNN) functionality, as well as Amazon SageMaker and its serverless inference feature. Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy ML models for any use case with fully managed infrastructure, tools, and workflows. Amazon OpenSearch Service is a fully managed service that makes it easy to perform interactive log analytics, real-time application monitoring, website search, and more.

Contrastive Language-Image Pre-Training (CLIP) is a neural network trained on a variety of image and text pairs. The CLIP neural network(s) is able to project both images and text into the same latent space, which means that they can be compared using a similarity measure, such as cosine similarity. 
You can use CLIP to encode your products’ images or description into embeddings, and then store them into a vector database. Then your customers can perform query in the database to retrieve products that they may have interest. To query the database, your customers need to provide input images or text, and then the input will be encoded with CLIP before sending to the vector database for KNN search. 

The vector database here plays the role of search engine. This vector database supports unify images and text-based search, which is particularly useful in the e-commerce and retail industries. One example of image-based search is your customers can search for a product by taking a picture, then query the database using the picture. Regarding to the text-based search, your customers can describe a product in free format text, and then use the text as a query.  The search results will be sorted by a similarity score (cosine similarity), if an item of your inventory is more similar to the query (an input image or text), the score will be closer to 1, otherwise the score will be closer to 0.  The top K products of your search results are the most relevant products in your inventory. 


### <ins> Solution overview </ins>

OpenSearch Service provides text-matching and embedding KNN-based search. We will use embedding KNN-based search in this solution. You can use both image and text as query to search items from the inventory. Implementing this unified image and test KNN-based search application consists of two phases:
- KNN reference index – In this phase, we pass a set of corpus documents or product images through CLIP model to encode them into embeddings. Text / image embeddings are a numerical representation of the corpus / image. You save those embeddings into a KNN index in the OpenSearch Service. The concept underpinning KNN is that similar data points exist in close proximity in the embedding space. As an example, text “a red flower”, text “rose” and an image of “red rose” are similar, so these text embeddings and image embeddings are close to each other in the embedding space.
- KNN index query – This is the inference phase of the application. In this phase, we submit a text search query or image search query through the deep learning model (CLIP) to encode as embeddings. Then, we use those embeddings to query the reference KNN index stored in the OpenSearch Service. The KNN index returns similar embeddings from the KNN embedding space. For example, if we pass the embedding of “a red flower” text, it would return the “red rose” embeddings as a similar item.
Next, let’s take a closer look at each phase, with the corresponding AWS architecture.

The solution uses the following AWS services and features:

![alt text](pictures/blog.drawio.png)
- Amazon S3 is used to store the raw product description text & images and image embedding generated by the SageMaker Batch Transform jobs. 
- A SageMaker Model is created from a pretrained CLIP model for batch and real-time inference. 
- SageMaker Batch Transform job is used to generate embeddings of product images.  
- SageMaker Serverless inference is used to encode query image and text into embeddings in real-time. 
- OpenSearch Service is the search engine to perform KNN-based search. 
- A query function is used to orchestrate encoding query and perform KNN-based search.
- SageMaker Studio Notebooks(not in the diagram) will be used as IDE to develop the solution.

### <ins> Instructions </ins>
#### I. Using CloudFormation to setup AWS SageMaker Studio and OpenSearch Domain

In the template `opensearch.yml`, it will create a OpenSearch Domain and grant your SageMaker Studio Execution role to use the domain.

In the template `sagemaker-studio-opensearch.yml`, it will create a new SageMaker Domain, a user profile in the Domain and a OpenSearch Domain. So you can use the StageMaker user profile to build this POC.

You can choose one of the templates to execute by following the steps listed below.

Step 1: Go to CloudFormation Service in your AWS console.
![alt text](pictures/CFN_UI.png)

Step 2: Upload a template to create a CloudFormation stack `clip-poc-stack`.

If you already have a SageMaker Studio running, you can use the template `opensearch.yml`.
![alt text](pictures/CFN_update_template.png)

If you don't have a SageMaker Studio at the moment, you can use the template `sagemaker-studio-opensearch.yml`. It will create a Studio Domain and user profile for you.


Step 3: Check the status of CloudFormation stack. It will take around 20 mins to finish the creation.
![alt text](pictures/CFN_result.png)

Once the stack is created, you can go to the SageMaker Console and click `Open Studio` to enter the Jupyter environment.
![alt text](pictures/studio-user.png)


If during the execution, the CloudFormation shows errors about the OpenSearch service linked role can not be found.
You need to create service-linked role by running `aws iam create-service-linked-role --aws-service-name es.amazonaws.com` in your AWS account.


#### II. Run the workflow

Please open the file `blog_clip.ipynb` with the SageMaker Studio and use `Data Science Python 3` kernel. You can execute cells from the start.

### <ins> Dataset </ins>

The [Amazon Berkeley Objects Dataset](https://registry.opendata.aws/amazon-berkeley-objects/) is used in the implementation. The dataset is a collection of 147,702 product listings with multilingual metadata and 398,212 unique catalogue images. We will only make use of the item images and item names in US English. For demo purposes we are going to use ~1,600 products.

## Running Costs

This section outlines cost considerations for running this demo. Completing the POC will deploy a OpenSearch Cluster and a SageMaker Studio which will cost less than $2 per hour. Noted: the price listed below is calculated using us-east-1 region. The cost varies from region to region. And the cost may change over time as well (the price here is recorded 2022-11-22). 

Further cost breakdowns are below.

- **OpenSearch Service** – Prices vary based on instance type usage and Storage cost. For more information, see [Amazon OpenSearch Service pricing](https://aws.amazon.com/opensearch-service/pricing/).  
  - The `t3.small.search` instance runs for approx 1 hour at \$0.036 per hour.

- **SageMaker** – Prices vary based on EC2 instance usage for the Studio Apps, Batch Transform jobs and Serverless Inference endpoints. For more information, see [Amazon SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/).

  - The `ml.t3.medium` instance for *Studio Notebooks* runs for approx 1 hour at \$0.05 per hour.
  - The `ml.c5.xlarge` instance for *Batch Transform* runs for approx 6 minutes at \$0.204 per hour.
  - The 6144 MB Serverless Inference for *inference* uses \$0.0001200 an second.
  
- **S3** – Low cost, prices will vary depending on the size of the models/artifacts stored. The first 50 TB each month will cost only $0.023 per GB stored. For more information, see [Amazon S3 Pricing](https://aws.amazon.com/s3/pricing/).
## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

