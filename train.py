from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# List of 100 prompt-completion pairs
data = [
    {"prompt": "Suggest decor for a rustic wedding in pink and white.", "completion": " Use blush pink and ivory roses, burlap table runners, mason jar centerpieces, wooden signage, and twinkle lights for a romantic, rustic ambiance."},
    {"prompt": "Suggest decor for a modern wedding in navy and gold.", "completion": " Opt for navy tablecloths, gold flatware, geometric candle holders, white roses, and sleek lounge furniture for a luxurious, modern vibe."},
    {"prompt": "Suggest decor for a bohemian baby shower in peach and cream.", "completion": " Use peach-toned florals, macrame hangings, pampas grass, wooden signage, and cream-colored tableware to create a soft boho atmosphere."},
    {"prompt": "Suggest decor for a tropical birthday party.", "completion": " Decorate with palm leaves, pineapples, flamingo props, bright florals, and tropical print tablecloths for a fun island vibe."},
    {"prompt": "Suggest decor for a classic black and white wedding.", "completion": " Use crisp white linens, black chair sashes, tall white floral arrangements, black candles, and classic chandeliers for an elegant setting."},
    {"prompt": "Suggest decor for a baby boy shower in blue and silver.", "completion": " Decorate with baby blue balloons, silver stars, moon-themed backdrops, cloud-shaped pillows, and silver streamers."},
    {"prompt": "Suggest decor for a 30th birthday in rose gold.", "completion": " Use rose gold foil balloons, sequined tablecloths, champagne flutes, metallic streamers, and a sparkle photo booth backdrop."},
    {"prompt": "Suggest decor for a sunflower-themed bridal shower.", "completion": " Decorate with sunflower centerpieces, yellow and green accents, rustic wooden trays, and personalized mason jars for drinks."},
    {"prompt": "Suggest decor for a garden wedding in lavender and sage.", "completion": " Use lavender bundles, sage green napkins, floral arches, vintage lace runners, and soft lighting in a botanical garden setting."},
    {"prompt": "Suggest decor for a baby girl shower in pink and gold.", "completion": " Include pink and gold balloons, a tutu table skirt, floral backdrops, glittery cupcake toppers, and golden 'Baby Girl' signage."},
    {"prompt": "Suggest decor for a Halloween-themed wedding.", "completion": " Incorporate black roses, candlelit pumpkins, deep burgundy linens, antique candelabras, and moody lighting for a spooky romantic vibe."},
    {"prompt": "Suggest decor for a pastel-themed baby shower.", "completion": " Use pastel balloons, baby animal cutouts, soft pink and mint tableware, cloud centerpieces, and a pastel rainbow cake."},
    {"prompt": "Suggest decor for a boho-chic wedding in terracotta and cream.", "completion": " Use terracotta pots, dried florals, neutral linens, fringe banners, and cozy floor cushions for a laid-back feel."},
    {"prompt": "Suggest decor for a royal blue and silver engagement party.", "completion": " Include silver lanterns, royal blue napkins, crystal centerpieces, and a navy and silver dessert table."},
    {"prompt": "Suggest decor for a jungle-themed birthday party.", "completion": " Use jungle leaf garlands, animal balloons, faux vines, and a green, brown, and yellow color palette with safari props."},
    {"prompt": "Suggest decor for a Valentine's Day wedding.", "completion": " Incorporate red roses, heart-shaped décor, romantic lighting, blush and red linens, and gold accents for a love-filled setting."},
    {"prompt": "Suggest decor for a winter wonderland baby shower.", "completion": " Use snowflake cutouts, icy blue and silver decorations, white faux fur table runners, and hanging crystal ornaments."},
    {"prompt": "Suggest decor for a tea party bridal shower.", "completion": " Decorate with floral teacups, pastel tablecloths, vintage tiered trays, lace doilies, and pink peonies."},
    {"prompt": "Suggest decor for a beach-themed wedding in turquoise and white.", "completion": " Use seashell centerpieces, turquoise table runners, driftwood signage, and white flowers with blue accents."},
    {"prompt": "Suggest decor for a glam 40th birthday party in black and gold.", "completion": " Use gold sequin backdrops, black tablecloths, gold cutlery, LED signage, and champagne towers."},
    {"prompt": "Suggest decor for a circus-themed first birthday.", "completion": " Include striped tents, popcorn boxes, red and yellow balloons, clown props, and carnival games."},
    {"prompt": "Suggest decor for a sunflower and mint green baby shower.", "completion": " Use sunflower bouquets, mint green streamers, polka-dot napkins, and wooden signage for a fresh look."},
    {"prompt": "Suggest decor for a Barbie-themed birthday party.", "completion": " Decorate with pink everything—balloons, backdrop, glitter streamers—and Barbie cutouts and themed desserts."},
    {"prompt": "Suggest decor for a minimalist wedding in white and beige.", "completion": " Use neutral-toned florals, white candles, simple tableware, linen runners, and minimal signage."},
    {"prompt": "Suggest decor for a dinosaur-themed birthday party.", "completion": " Use green balloons, dinosaur footprints, jungle vines, dino figurines, and fossil-like table centerpieces."},
    {"prompt": "Suggest decor for a rainbow baby shower.", "completion": " Include rainbow balloon arches, multicolored table settings, clouds and raindrop decor, and rainbow-themed cupcakes."},
    {"prompt": "Suggest decor for a vintage garden wedding.", "completion": " Incorporate antique furniture, lace tablecloths, birdcages, roses, and soft pastel tones."},
    {"prompt": "Suggest decor for a Great Gatsby-inspired wedding.", "completion": " Use gold and black art deco details, crystal chandeliers, feathers, jazz-era fonts, and elegant bar setups."},
    {"prompt": "Suggest decor for a wild west-themed birthday.", "completion": " Use hay bales, cowboy hats, wooden signage, red bandanas, and cactus centerpieces."},
    {"prompt": "Suggest decor for a unicorn-themed birthday party.", "completion": " Include pastel balloons, unicorn horns, glittery cupcakes, rainbow decor, and iridescent streamers."},
    {"prompt": "Suggest decor for a rustic autumn wedding.", "completion": " Use pumpkins, burgundy florals, golden leaves, plaid blankets, and wooden barrels for a cozy fall aesthetic."},
    {"prompt": "Suggest decor for a Parisian baby shower.", "completion": " Include Eiffel Tower centerpieces, macarons, blush pink and black color palette, and striped backdrops."},
    {"prompt": "Suggest decor for a spa-themed birthday party.", "completion": " Use white robes, candles, cucumber-infused water, soft towels, and floral-scented decor elements."},
    {"prompt": "Suggest decor for a neon glow birthday party.", "completion": " Use glow sticks, blacklights, neon posters, LED signs, and highlighter-colored balloons."},
    {"prompt": "Suggest decor for a celestial-themed wedding.", "completion": " Incorporate starry backdrops, moon and star cutouts, navy and silver accents, and dreamy lighting."},
    {"prompt": "Suggest decor for a garden brunch bridal shower.", "completion": " Use floral centerpieces, vintage teapots, pastel napkins, mimosa stations, and outdoor seating."},
    {"prompt": "Suggest decor for a peacock-themed wedding.", "completion": " Include jewel tones like teal, purple, and gold, feather accents, ornate details, and dramatic florals."},
    {"prompt": "Suggest decor for a Frozen-themed birthday party.", "completion": " Use icy blue and white balloons, snowflake cutouts, Elsa cutouts, and glittery table settings."},
    {"prompt": "Suggest decor for a Hollywood red carpet party.", "completion": " Use red carpets, gold stars, film reel props, velvet ropes, and paparazzi-style backdrops."},
    {"prompt": "Suggest decor for a fiesta-themed engagement party.", "completion": " Include colorful papel picado, sombreros, cacti centerpieces, maracas, and vibrant tablecloths."},
    {"prompt": "Suggest decor for a safari baby shower.", "completion": " Use animal prints, jungle foliage, soft greens and browns, stuffed animals, and explorer-themed props."},
    {"prompt": "Suggest decor for a nautical-themed wedding.", "completion": " Incorporate anchors, navy and white stripes, rope details, seashell centerpieces, and lanterns."},
    {"prompt": "Suggest decor for a woodland-themed baby shower.", "completion": " Use forest animal cutouts, tree stump cake stands, pinecones, ferns, and neutral color schemes."},
    {"prompt": "Suggest decor for a music-themed birthday party.", "completion": " Use vinyl record decor, musical note cutouts, instrument props, and black and red color palettes."},
    {"prompt": "Suggest decor for a lemon-themed bridal shower.", "completion": " Include lemon centerpieces, yellow and white linens, citrus drinks, and greenery garlands."},
    {"prompt": "Suggest decor for a disco-themed wedding afterparty.", "completion": " Use disco balls, silver streamers, neon lighting, funky dance floor, and retro decor."},
    {"prompt": "Suggest decor for a classy champagne-themed birthday brunch.", "completion": " Use blush tones, gold-rimmed glasses, champagne bottle centerpieces, and elegant floral touches."},
    {"prompt": "Suggest decor for a superhero kids' birthday party.", "completion": " Use comic-style banners, city skyline backdrops, superhero capes, bold red, blue, and yellow colors."},
    {"prompt": "Suggest decor for a romantic Valentine's baby announcement.", "completion": " Include heart garlands, pink and red tones, soft lighting, and a 'Love Grows' sign with ultrasound photo."},
    {"prompt": "Suggest decor for a vintage movie-themed wedding.", "completion": " Use old film reels, black and white photos, classic marquee signs, red curtains, and gold accents for a timeless feel."},
    {"prompt": "Suggest decor for a lavender and lemon summer bridal shower.", "completion": " Include lavender sprigs, lemon centerpieces, yellow linens, and soft purple florals for a fresh and elegant ambiance."},
    {"prompt": "Suggest decor for a travel-themed baby shower.", "completion": " Use mini suitcases, globes, paper airplanes, map backdrops, and passport-inspired invites and favors."},
    {"prompt": "Suggest decor for a masquerade-themed wedding reception.", "completion": " Decorate with ornate masks, purple and gold accents, candlelight, feathers, and velvet table settings."},
    {"prompt": "Suggest decor for a green and gold engagement party.", "completion": " Use emerald green table runners, gold cutlery, tropical leaves, and string lights for a lush celebration."},
    {"prompt": "Suggest decor for a unicorn baby shower.", "completion": " Use pastel rainbow balloons, unicorn centerpieces, gold stars, glittery signage, and cloud-shaped pillows."},
    {"prompt": "Suggest decor for a carnival-themed wedding.", "completion": " Incorporate red and white stripes, popcorn stands, vintage signs, fairy lights, and playful game booths."},
    {"prompt": "Suggest decor for a floral and lace baby shower.", "completion": " Use soft lace runners, pink roses, vintage teacups, pearl accents, and pastel balloons for a delicate theme."},
    {"prompt": "Suggest decor for a retro 80s-themed birthday party.", "completion": " Use neon colors, cassette tapes, Rubik’s cubes, checkered patterns, and disco balls for a nostalgic vibe."},
    {"prompt": "Suggest decor for a moon and stars-themed baby shower.", "completion": " Include crescent moon cutouts, gold star garlands, navy blue accents, and dreamy lighting."},
    {"prompt": "Suggest decor for a cherry blossom spring wedding.", "completion": " Use cherry blossom branches, pale pink linens, delicate string lights, and white lanterns for a romantic atmosphere."},
    {"prompt": "Suggest decor for a boho forest engagement party.", "completion": " Use earthy tones, macrame hangings, wooden elements, fairy lights, and wildflowers for a cozy natural feel."},
    {"prompt": "Suggest decor for a breakfast at Tiffany’s bridal shower.", "completion": " Include Tiffany blue accents, pearl details, elegant black tableware, and Audrey Hepburn silhouettes."},
    {"prompt": "Suggest decor for a garden fairy-themed baby shower.", "completion": " Use toadstool props, fairy lights, flower garlands, pastel colors, and whimsical wings for guests."},
    {"prompt": "Suggest decor for a space-themed kids’ party.", "completion": " Include rocket centerpieces, glow-in-the-dark stars, space helmets, galaxy balloons, and alien cutouts."},
    {"prompt": "Suggest decor for a glam pink and black birthday party.", "completion": " Use hot pink and black balloons, glittery tablecloths, feather boas, and dramatic lighting."},
    {"prompt": "Suggest decor for a minimal blush and ivory wedding.", "completion": " Use soft blush roses, ivory linens, clear acrylic signage, gold accents, and minimal greenery."},
    {"prompt": "Suggest decor for a whimsical butterfly-themed baby shower.", "completion": " Include butterfly garlands, pastel flowers, tulle drapes, and soft lighting for a dreamy atmosphere."},
    {"prompt": "Suggest decor for a cozy cabin winter wedding.", "completion": " Use pinecones, plaid accents, candlelight, fur throws, and wood details for a warm rustic feel."},
    {"prompt": "Suggest decor for a rainbow art-themed birthday.", "completion": " Use colorful paint splatters, easel stations, paintbrush favors, and rainbow cupcakes."},
    {"prompt": "Suggest decor for a champagne and blush bridal brunch.", "completion": " Use champagne towers, blush florals, rose gold cutlery, and soft background music."},
    {"prompt": "Suggest decor for a camping-themed kids’ birthday party.", "completion": " Use tents, faux campfires, lanterns, woodland animals, and s’mores stations."},
    {"prompt": "Suggest decor for a disco roller skate party.", "completion": " Use glitter balls, retro colors, skate-shaped balloons, neon lights, and funky dance tracks."},
    {"prompt": "Suggest decor for a New Year’s Eve wedding.", "completion": " Use gold confetti, black and white tableware, countdown clocks, and sparklers for a festive evening."},
    {"prompt": "Suggest decor for a vintage library-themed wedding.", "completion": " Use old books, antique keys, calligraphy signage, candlelight, and dark wood accents."},
    {"prompt": "Suggest decor for a beach bonfire engagement party.", "completion": " Include tiki torches, driftwood signs, low tables with cushions, lanterns, and a firepit centerpiece."},
    {"prompt": "Suggest decor for a pink lemonade summer baby shower.", "completion": " Use lemon and pink garlands, citrus centerpieces, pink drinks, and yellow napkins."},
    {"prompt": "Suggest decor for a navy and blush outdoor wedding.", "completion": " Use navy tablecloths, blush roses, wooden chairs, and gold candleholders for a classic touch."},
    {"prompt": "Suggest decor for a winter baby shower in blue and white.", "completion": " Use snowflakes, blue balloons, silver accents, and soft white fur textures."},
    {"prompt": "Suggest decor for a nautical-themed first birthday.", "completion": " Use anchors, red and navy accents, sailor hats, rope decor, and boat-themed snacks."},
    {"prompt": "Suggest decor for a 90s throwback birthday party.", "completion": " Include boom boxes, slime green and purple colors, CDs, and retro cartoons for a nostalgic vibe."},
    {"prompt": "Suggest decor for a red and gold romantic wedding.", "completion": " Use red roses, gold candlesticks, velvet tablecloths, and ornate tableware for a passionate look."},
    {"prompt": "Suggest decor for a rustic baby shower with neutral tones.", "completion": " Use beige and white decor, burlap runners, dried florals, and wooden crates."},
    {"prompt": "Suggest decor for a soft blue and cream beach wedding.", "completion": " Include seashells, cream linens, soft blue flowers, and driftwood centerpieces."},
    {"prompt": "Suggest decor for a Halloween baby shower.", "completion": " Use mini pumpkins, black and orange balloons, spider web accents, and ghost-themed treats."},
    {"prompt": "Suggest decor for a mermaid-themed birthday party.", "completion": " Use teal and purple balloons, shell cutouts, sequins, and under-the-sea backdrops."},
    {"prompt": "Suggest decor for a romantic garden dinner proposal.", "completion": " Use fairy lights, rose petals, white drapes, candlelit pathways, and soft acoustic music."},
    {"prompt": "Suggest decor for a Gatsby glam bridal shower.", "completion": " Use art deco patterns, gold and black accents, champagne flutes, and feathers."},
    {"prompt": "Suggest decor for a cactus-themed baby shower.", "completion": " Use mini cacti, terra cotta pots, green and pink color palette, and succulent-themed cookies."},
    {"prompt": "Suggest decor for a science-themed birthday party.", "completion": " Use beaker props, periodic table decor, slime stations, and lab coat dress-up."},
    {"prompt": "Suggest decor for a minimal monochrome wedding.", "completion": " Use all-white decor with black text, geometric accents, and sleek florals for a clean modern look."},
    {"prompt": "Suggest decor for a circus-themed bridal shower.", "completion": " Use elegant circus stripes, gold accents, pink and red florals, and whimsical dessert stands."},
    {"prompt": "Suggest decor for a spring-themed engagement party.", "completion": " Use pastel flowers, butterfly garlands, floral arches, and outdoor lounge seating."},
    {"prompt": "Suggest decor for a Disney princess baby shower.", "completion": " Include tiaras, pastel colors, castle backdrops, and themed table settings for each princess."},
    {"prompt": "Suggest decor for a red rose and pearl wedding.", "completion": " Use lush red roses, pearl garlands, crystal chandeliers, and gold-rimmed glasses."},
    {"prompt": "Suggest decor for a boho sunset wedding.", "completion": " Use warm oranges and pinks, dreamcatchers, dried palms, floor rugs, and string lights."},
    {"prompt": "Suggest decor for a jungle safari bridal shower.", "completion": " Use green leaves, gold animal figurines, bamboo trays, and wildflower arrangements."},
    {"prompt": "Suggest decor for a Paris-themed birthday party.", "completion": " Include Eiffel Tower props, pink and black decor, French pastries, and striped table runners."}
]

# Format the dataset
formatted_data = [{"text": item["prompt"] + item["completion"]} for item in data]
dataset = Dataset.from_list(formatted_data)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training settings
training_args = TrainingArguments(
    output_dir="./decor-gpt2",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./decor-gpt2")
    tokenizer.save_pretrained("./decor-gpt2")