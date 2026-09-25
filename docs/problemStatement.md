# Problem Statement: Google Photos Discovery Engine

## Role & Context

You are a **Product Manager on the Core Experience team at Google Photos**.

Over years of usage, users accumulate thousands of photos, videos, screenshots, documents, and other visual memories in Google Photos.

While users can easily search when they know what they are looking for, **retrieval becomes much harder when memory is incomplete**.

### Example Scenarios

> *"That small café we went to during our Goa trip."*

> *"The picture of the medicine I took when I was sick last year."*

The user **knows the photo exists** — but may not remember:
- When it was taken
- Where it was taken
- Which album it belongs to
- The exact words needed to search for it

---

## Strategic Goal

> **Increase the percentage of users who successfully retrieve a photo they remember but cannot precisely describe when they start searching.**

The challenge is **not** to improve search in general. The focus is specifically on vague/incomplete memory-based retrieval.

**Core Task**: Understand how people remember old visual information, where the existing retrieval experience breaks down, and identify an opportunity that can meaningfully improve successful retrieval.

---

## Part 1: Build an AI-Powered Discovery Engine

Before proposing any solution, build an AI-powered system that **analyzes user feedback and conversations about photo retrieval at scale**.

### Data Sources to Analyze

- Google Play Store reviews
- App Store reviews
- Reddit discussions
- Google Photos community / support discussions
- Social media conversations
- YouTube comments
- Forums and other relevant public discussions

### Key Questions to Uncover

- What kinds of old photos do users struggle to retrieve?
- What information do people actually remember about a photo?
- What information have they forgotten?
- How do users formulate searches when their memory is incomplete?

The workflow should go **beyond summarizing reviews or performing sentiment analysis**. It should enable identification and comparison of different retrieval problems and opportunity areas using evidence from real users.

---

## Part 2: Break Down the Business Metric

Decompose the metric:

> **Successful retrieval of vaguely remembered photos**

...into the relevant **user behaviors** and **product outcomes**.

Develop a decomposition based on understanding of the product and the evidence surfaced by the discovery engine.

### Diagnostic Questions to Consider

| Question | What It Probes |
|---|---|
| Is the user unable to express what they remember? | Expression / articulation gap |
| Does Google Photos fail to understand the clues they provide? | Intent understanding gap |
| Are potentially relevant results difficult to evaluate? | Result presentation gap |
| Does the user struggle to refine an unsuccessful search? | Recovery / iteration gap |

Use this decomposition to identify **where the greatest opportunities may exist**.
