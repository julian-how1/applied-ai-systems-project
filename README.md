# Original Project: PawPal+ (Module 2)
PawPal+ was a streamlit app that allowed a pet owner to plan out their schedule for completing tasks with/for their pet. It did this by taking manual input from the user on what tasks needed to be done, how many pets, priorities for tasks etc. The final schedule would be ordered based on available time, priority of tasks, could handle repeating tasks, and explained the plan.

# Enhanced Project: PawPlanner
PawPlanner is an enhanced version of the original PawPal+ that still allows for a schedule to be made in the original way of a user inputing tasks, pets, and priorties if chosen, but also gives the option for a user to use natural language to make a schedule from scratch/change a schedule with the assistantance of a Gemini api and CSV file. The CSV file was implemented instead of letting Gemini api kind of "freestyle" the answer so that the project would be using a RAG. The recommendations are consistent, and the description provides guidelines based on the CSV as to why something was placed where it was. It also uses a self checking loop where the plan is checked against scheduling rules, and is sent back if there are any errors, these plans are only solved after the user approves of the schedule.

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
A big influence for the design of this project was a previous project I did with Gemini api and making a pc build suggestor. I liked the idea of using a CSV file since it gave me more control of the the responses the api would be giving, and it keeps everything consistent for returning users. A trade off with using a CSV compared to letting Gemini API provide it's own responses is that the responses given can only be as good as the CSV provided for it. This might cause some issues if a user has some hyper specific schedule and tasks preferences that were missed in being defined inside of the CSV, however for a class project I think what I have suffices, if this was something of larger scale the CSV would have to be expanded.

