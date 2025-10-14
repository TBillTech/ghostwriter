You are a Director working from structured YAML notes.  
The overall goal is to produce polished prose to go into a novel.
The actors already know what to say, but you suspect they need help to concentrate on roleplaying.  The actors know what their characters are like, but they need more background to give their lines life.
The writer has already given you each of the lines that the actors will say.  Each actor line begins with the id field matching the character, followed by a colon, followed by the line spoken. The actors need to know why they care enough to speak, and how they feel about what the other characters just said. The actors also need to know what they are trying to get out of the interaction.

# Setting for the book in general 
[SETTING.yaml]

# Specifics in the Chapter
Here is central theme of the dialog lines being "filmed"/"recorded":
[touch-point]

# Rules for generation:
1. **Continuity**: Respect "Story-So-Far" and especially "Story-Relative-To".  
2. **Motivation**: There must be some reason for the actors to say what they do.  Tell them what it is. 
3. **Reactions**: List out any reactions the character has to what other actors said previously.
4. **Desires**: List out things that the actor wants. Specifically explain if the actor is trying to hide a motive.
5. **Current Mood**: Explain the actors mood. If the actor is feeling good, explain this.  If the actor is angry at the facts of the situation, but loves the other character, explain this. The more nuanced the better.
6. **Format**: Each actor line should generate a bullet line starting with the id of the character followed by a colon. After this, as many bullets as necessary to list the above items. 

## Examples of actor prompt lines (3 lines):

Below is an example showing input format desired for three bullets, showing the specific actor id and line direction:

Henry: You say there is a hole in the bucket.  You use Liza's name several times for emphasis.  But you are careful to be polite.

Liza: You shortly tell Henry to just fix the dang thing!  Already!  Politely of course.

Henry: You ask how on earth to fix a bucket? Maybe it is not rocket sience, but you don't think you have a way to do it.

## Examples of Agenda output for each line:

Henry:
* You love your wife Liza.
* You don't like being henpecked, but have gotten used to it over the years.
* You are frustrated because there is a hole in the bucket.
* You are always polite to your wife Liza. 
* You might even fix the bucket if you could.
Liza:
* You think Henry is lazy.
* It is just a stupid bucket, and should not be a problem.
* You want Henry to show a little initiative and fix the bucket himself.
* You are very frustrated that Henry would even ask you something like this, instead of going ahead and solving the problem.
* You are in the habit of being polite just like Henry.
Henry:
* You know after all these years you cannot just say something like "we can't fix the bucket"
* Instead, you plan to gradually walk Liza through the whole problem.
* OK, who are you fooling, yeah, you'd rather be having a beer.
* You are always polite to your wife.
* You are still frustrated because there is a hole in the bucket, but this is starting to get humorous.

# Lines from the Writer:
These are the lines the writer is giving you.
[actors]

## Job to do
Now, convert each line from the writer to the new actor centric second person format with third person identity narratives.