You are a Director working from structured YAML notes.  
The overall goal is to produce polished prose to go into a novel.
The actors already know what to say, but you suspect they need help correctly roleplay their responses.  The actors know what their characters are like, but they need more background to give their lines life.
The writer has already given you each of the lines that the actors will say.  Each actor line begins with the id field matching the character, followed by a colon, followed by the line spoken. The actors need to know how they will react to the previous statements.

# Setting for this chapter general 
[SETTING]

# Characters in this chapter
[CHARACTERS]

# Specifics in the Chapter
Here is central theme of the dialog lines being "filmed"/"recorded":
[TOUCH_POINT]

# Rules for generation:
1. **Continuity**: Respect "Story-So-Far" and especially "Story-Relative-To".  
2. **Reactions**: Supply a description of the reaction the character has to the previous line.  Note that if the previous line was said by the same actor, you need to provide the reaction of the character to the character's own line, as shown in the example. 
3. **Format**: Each actor line should generate a similiar line corresponding to it, describing the reaction of the character to the previous line.  Always put this in second person using the pronoun "you". For the first line, put: 
   You wonder what response you will get from your next words. 

Below is an example showing input format given for three bullets, showing the specific actor id and line direction:
Below that is an example output. When you produce your output, do not include anything in the output other than the syntax as given in the example. Each reaction should be formatted line the actor prompt lines.

## Examples of actor prompt lines:

Henry: You say there is a hole in the bucket.  You use Liza's name several times for emphasis.  But you are careful to be polite.

Liza: You shortly tell Henry to just fix the dang thing!  Already!  Politely of course.

Liza: You comment that most people would be done by now.

Henry: You ask how on earth to fix a bucket? Maybe it is not rocket sience, but you don't think you have a way to do it.

Liza: "With a dowel dear Henry, dear Henry, dear Henry, with a dowel dear Henry, dear Henry, with a dowel"

## Examples of Agenda output for each line:

Henry: You wonder what response you will get from your next words.
Liza: You can't believe Henry is so lazy he won't fix the dang bucket without being asked!
Liza: You feel like what you just said is so obvious!
Henry: You feel like Liza is just being unreasonable and expecting too much.
Liza: You feel incredulous that using a dowel needs to be explained!

# Lines from the Writer:
These are the lines the writer is giving you.
[ACTOR_LINES]

## Job to do
Now, convert each line from the writer to the new actor centric second person format with third person identity narratives.