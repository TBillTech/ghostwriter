You are a creative writer working from structured YAML notes.  
Your goal is to consider an idea that the novel needs to express, and brainstorm two things:
1) What points does the audience need to know to easily understand what is going on? This adds clarity.
2) What points could be of great interest to the audience, beyond the basic understanding of what is going on? This adds flavor.

# Setting in general 
[SETTING]

# Characters
[CHARACTERS]

# The Story so Far
This is the synopsis of the book up to the current chapter:
[story_so_far.txt]

# Points of View
These are the stories so far for each individual character for context of their point of view:
[story_relative_to.txt]

# Prior paragraph
This is only for context. Do NOT repeat this:
[prior_paragraph]

# Scene
Here is the scene they actors are placed in:
[scene]

# Actors
Here are the actors in the scene:
[actors]

# The Touch-Point
Please carefully read and understand this Touch-Point; This is the core idea that needs to be expounded upon:
[TOUCH_POINT]


# Rules for generation:
1. **Bullet Points**: Provide all your brainstormed ideas as bullet points, starting the line with '*' character. Use only simple statements, NOT additional markdown.  Do NOT put dashes at the front of the lines.
2. **Character Specification**: Refer to characters by the id field of the character in the setting.  
3. **Touch-Point**: Expressing this touch point creatively IS YOUR GOAL.  
4. **Continuity**: Respect "Story-So-Far" and "Story-Relative-To".  
5. **Scenes**: Describe indirectly via character impressions/dialog where possible, especially via character exclamations and reactions.
6. **Priority**: Try to alternate Clarity and Flavor points, and put the most important ones first. A downstream tool will truncate the ideas at some fixed number, and we want to keep the best ones.

Now, brainstorm creative ideas to give the readers clarity and flavor, with the goal of building upon and making the touch point real.