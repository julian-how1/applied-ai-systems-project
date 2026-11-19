# Original Project: PawPal+ (Module 2)
PawPal+ was a streamlit app that allowed a pet owner to plan out their schedule for completing tasks with/for their pet. It did this by taking manual input from the user on what tasks needed to be done, how many pets, priorities for tasks etc. The final schedule would be ordered based on available time, priority of tasks, could handle repeating tasks, and explained the plan.

# Enhanced Project: PawPlanner
PawPlanner is an enhanced version of the original PawPal+ that still allows for a schedule to be made in the original way of a user inputing tasks, pets, and priorties if chosen, but also gives the option for a user to use natural language to make a schedule from scratch/change a schedule with the assistantance of a Gemini api and CSV file. The CSV file was implemented instead of letting Gemini api kind of "freestyle" the answer so that the project would be using a RAG. The recommendations are consistent, and the description provides guidelines based on the CSV as to why something was placed where it was. It also uses a self checking loop where the plan is checked against scheduling rules, and is sent back if there are any errors, these plans are only solved after the user approves of the schedule.

# Architecture Overview: 
The system diagram shows PawPlanner as four connected groups: the Streamlit app the user interacts with, the AI agent, the data and services it relies on, and the testing and evaluation tools. Data flows left to right: the pet owner's plain-English request goes from the AI Planner page into the agent, which moves through five steps: Interpret, Retrieve, Propose, Check, and Explain. During Retrieve, the agent pulls matching care guidelines from the CSV knowledge base, and during Propose and Explain it exchanges prompts and replies with the Gemini API. The Check step tests every proposal against the original scheduler, and the loop arrow back to Propose shows that a plan which breaks the rules is returned to the AI to fix, up to three times. The finished plan then reaches the orange Human Review diamond, where the owner chooses to apply or discard it before anything is saved. The green Tester and Evaluator boxes show that the agent is also checked by automated tests and real-AI evaluation scenarios, while the "Build it myself" path shows that the original manual workflow still works without the AI.


# Setup instructions
1. Requirements
  - Python 3.10 or newer
  - A free gemini API key is needed to run this project (you can get one from Google Ai Studio)

2. Create a virtual environment and install dependencies
  - py -m venv .venv
  - .venv\Scripts\Activate.ps1
  - pip install -r requirements.txt

3. Add your API key
  - Copy .env.example to a file called .env and paste key after GEMINI_API_KEY = 

4. Run the app
  - streamlit run app.py

5. Run the test
  - pytest

6. Evaluate AI performance
  - python evaluate_ai.py
  
# Sample Interactions

These are real outputs from the AI Planner (model: `gemini-3.1-flash-lite`). AI responses can vary slightly from run to run.

## Example 1: Building a schedule from scratch

**Input** (no pets or tasks set up yet):
> I have a 3 year old dog named Rex. I'm free for 90 minutes starting at 7am. He needs a 30 minute walk, breakfast at 7:30, and his allergy pill with breakfast.

**Care guidelines retrieved from the CSV:** G32 Medication with food, G04 Breakfast, G01 Walk, G33 Medication, G05 Dinner

**What the AI understood:**
> The owner wants to add a 3-year-old dog named Rex to the schedule, with a 30-minute walk, breakfast at 07:30, and an allergy pill immediately following breakfast.

**Proposed changes** (passed every scheduling check on the first round):
- Set day start to 07:00
- Set available time to 90 min
- Added pet Rex (dog, age 3)
- Added 'Walk' for Rex (30 min, medium)
- Added 'Breakfast' for Rex (10 min, high at 07:30)
- Added 'Allergy pill' for Rex (5 min, high at 07:40)

**Resulting schedule:**

| Time | Pet | Task | Priority |
|---|---|---|---|
| 7:00 AM – 7:30 AM | Rex | Walk | medium |
| 7:30 AM – 7:40 AM | Rex | Breakfast | high |
| 7:40 AM – 7:45 AM | Rex | Allergy pill | high |

**AI explanation:**
> I have updated your schedule to include Rex's morning routine within your 90-minute window. Your plan now starts at 07:00 with a 30-minute walk, followed by breakfast at 07:30. To ensure Rex receives his allergy pill right after his meal, I have pinned it to 07:40. This schedule respects your priorities by pinning the high-priority feeding and medication tasks to your preferred times. There are no skipped tasks or warnings to report.

The AI's notes cited the guidelines it used ("Used G04, G01, and G32"). Guideline G32 says to pin medication to the minute a meal ends, because the scheduler can't run two tasks at once. That's why the pill landed at 7:40 instead of clashing with breakfast at 7:30.

## Example 2: Fixing an existing schedule

**Starting schedule** (two pets, Buddy the dog and Whiskers the cat, entered manually). Note the clash at 5:30 PM:
```
07:00-07:15  Litter box cleanup for Whiskers (15 min) - medium priority
07:30-08:00  Morning walk for Buddy (30 min) - high priority
08:15-08:25  Breakfast for Buddy (10 min) - high priority
09:00-09:05  Thyroid meds for Whiskers (5 min) - high priority
17:30-17:35  Ear drops for Buddy (5 min) - high priority
17:30-17:40  Dinner for Whiskers (10 min) - high priority
18:00-18:30  Evening walk for Buddy (30 min) - medium priority
Skipped: Brushing for Whiskers [low] - needs 20 min, only 15 min left
Warning: Ear drops (Buddy) and Dinner (Whiskers) both start at 17:30
```

**Input:**
> My evening feels crammed. Can you fix any clashes in the schedule?

**What the AI understood:**
> The user wants to resolve the scheduling conflict at 17:30 where Buddy's ear drops and Whiskers' dinner overlap.

**Proposed changes** (one minimal edit, passed every scheduling check on the first round):
- Edited 'Ear drops' for Buddy: preferred_time=17:40

**Resulting evening:**

| Time | Pet | Task | Priority |
|---|---|---|---|
| 5:30 PM – 5:40 PM | Whiskers | Dinner | high |
| 5:40 PM – 5:45 PM | Buddy | Ear drops | high |
| 6:00 PM – 6:30 PM | Buddy | Evening walk | medium |

**AI explanation:**
> I have adjusted your evening schedule to resolve the conflict at 17:30. Buddy's ear drops have been moved to 17:40, allowing Whiskers to have dinner at 17:30 without any overlap. This plan prioritizes your high-priority feeding and medication tasks while keeping your evening manageable. Please note that brushing for Whiskers has been skipped because it requires 20 minutes and only 15 minutes remain in your schedule.

The AI changed only the one task causing the clash and left the rest of the day alone. The validator also flagged the skipped brushing task, and the explanation passed that warning on to the user.

# Design Decisions
A big influence for the design of this project was a previous project I did with Gemini api and making a pc build suggestor. I liked the idea of using a CSV file since it gave me more control of the the responses the api would be giving, and it keeps responses consistent for the same types of request. A trade off with using a CSV compared to letting Gemini API provide it's own responses is that the responses given can only be as good as the CSV provided for it. This might cause some issues if a user has some hyper specific schedule and tasks preferences that were missed in being defined inside of the CSV, if it can't be found in the CSV it will have to fall back to the general info from Gemini. However, for a class project I think what I have suffices, if this was something of larger scale the CSV would have to be expanded.

# Testing Summary
- All 68 automated test pass, they used a fake AI so they don't need an API key to run
- 8 of 8 Gemini evaluations pass validity and usefullness checks
- With Rag vs. without RAG on creating scenarios (3/3 vs 2/3 successes)
## What didn't work
- Gemini's free tier kept getting overload and provided error code 503, before fixing with backup models the request would hang for 5+ minutes or return an error
- First rag attempt failed 2/3 scenarios making 2 bugs known: a timeout error, and a retrival in the users input on a preference, both ended up being fixed


# Reflection Summary (more in model card)
Working on this project taught me how important having back up plans are for your projects, since originally when trying to test the responses of the api I kept getting error "503 high demand". Claude showed me how to allow the api to use backup Gemini models, which greatly reduced the time that it took for a response to be generated, and completely prevented the timeouts and error codes from happening. Another thing it taught me was how even though coding with AI makes things easier and faster that it's still really important to read everything that your coding assistant is doing, and you might have to step in and tell it to stop. There were a handful of times where I instructed it to do one task, but it just kept going on and on, and I had to stop it and explained that it felt like I was losing control of what was being made because it was getting creative without me. Some future improvements for this project would be getting actual vet-reviewed guidelines, a larger CSV, and a better system rather than keyword matching.
