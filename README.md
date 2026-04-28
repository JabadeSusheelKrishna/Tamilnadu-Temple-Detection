# SMAI - Assignment - 3

```
topic: T12.4 - Temples of Tamil Nadu
team: Naa Chaavu Nen Chastha
members: Susheel Krihna Jabade - 2022101006
```

**The Problem**: A tourist points their phone at a monument; the app names it, gives history, opening hours, and ticket prices.

**What you will build**: 
- Streamlit app: 
    - upload photo
    - predicted monument 
    - history paragraph 
    - visit info card 
    - open in Google Maps.
- CLIP zero-shot on `15` famous TN temples; 
- images from Wikimedia Commons [Hindu temples in Tamil Nadu.](https://commons.wikimedia.org/wiki/Category:Hindu_temples_in_Tamil_Nadu)

**Skills**: CLIP zero-shot or light fine-tuning, Wikipedia scraping for metadata.

**Resources**: 
- For zero-shot: `openai/clip-vit-base-patch32` ; CLIP needs no training data, just a class-
name list.
- Metadata (history, hours, tickets): scrape Wikipedia or ask Gemini once and cache as JSON.

---

# My Plan: 

- first we need to get the 15 temples images data from the wikipedia. also need to gather meta data!
- we need to implement multiple versions of backend paths! representing different strategies!
- in one version, we can do baseline one! just give temple names as text and image directly as input and get the probability of each temple! also we need to have a negative class for Non temple images! this is so that our model will not give temple name for any image which is not even temple!
- in one version, we can do better prompt engineering instead of just giving temple names
- in one version, instead of normally converting any image to 228 X 228 pixel, we can focus on the region where the architecture exists!
- in one version, we can edit the image (its brightness, contrast, sharpness) to some normalized version.
- in one version, as we get image embedding and texts embeddings, instead of just using a softmax, we can pass it again to a simple FFNN with 1 hidden layer to capture hidden relations! remember that we need to do training for it!