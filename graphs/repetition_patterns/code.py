import matplotlib.pyplot as plt
import seaborn as sns

def gen_lyric_points(lyrics):
    """takes list of words and finds indexes that correspond to repetitions of the words at some point
    """
    x = []; y = []
    n_lyrics = len(lyrics)
    for ix, word_target in enumerate(lyrics):
        for iy, word in enumerate(lyrics):
            if word_target == word:
                x.append(ix)
                # flip the vertical axis
                y.append(n_lyrics - iy)
    return x,y 


def main():
    """ Code by omgardner. I didn't use much of seaborn's functionality, but there are some styling functions used. The majority is `matplotlib.pyplot`.
    """
    sns.set_style('white')

    # hardcoded for brevity
    lyrics = "love love love love love love love love love theres nothing you can do that cant be done nothing you can sing that cant be sung nothing you can say but you can learn how to play the game its easy theres nothing you can make that cant me made no one you can save that cant be saved nothing you can do but you can learn how to be you in time its easy all you need is love all you need is love all you need is love love love is all you need love love love love love love love love love all you need is love all you need is love all you need is love love love is all you need theres nothing you can know that isnt known nothing you can see that isnt shown theres nowhere you can be that isnt where youre meant to be its easy all you need is love all you need is love all you need is love love love is all you need all you need is love all together now all you need is love everybody all you need is love love love is all you need love is all you need love is all you need love is all you need"
    song_name = "all you need is love"
    artist_name = "the beatles"
    
    # create subplot to adjust figsize
    fig, ax = plt.subplots(figsize=(30,30))

    # x,y lyric repetitions converted to coordinates, based on index position in '2x2 matrix'

    lyrics = lyrics.split() # remove this for some wacky stuff
    x,y = gen_lyric_points(lyrics)

    # scatterplot, the iterable for `c` variable creates a diagonal number pattern, which the cmap maps to its colours
    ax.scatter(x, y, c=[xx-yy for xx,yy in zip(x,y)], cmap='spring', s=5)

    # change background color
    ax.set_facecolor('#000000')

    # set title, family and fontsize are parsed to **kwargs : Text properties
    ax.set_title(f"Repetition pattern in: \"{song_name}\" by \"{artist_name}\"",
                family="Garamond", 
                fontsize=20
    )

    # lyrics along y axis
    ax.set_yticklabels(lyrics, family='Calibri')

    # show all lyrics, but reverse existing order so it reads top-to-bottom
    ax.set_yticks(range(len(lyrics),0,-1))

    # remove x ticks
    ax.set_xticklabels([])

    # hey its me
    ax.set_xlabel("omgardner")

    # remove grid
    ax.grid(False)

    # remove 'spines': the edges of a graph, sns.despine removes top and right by default
    sns.despine(ax=ax, left=True, bottom=True)

    # margin between spines and plot content, leaves slight gap
    ax.margins(x=0.005, y=0.005)

    # dynamic formatting of filename
    fp = f"{'_'.join(song_name.split())}_BY_{'_'.join(artist_name.split())}_example.png"

    # bbox_inches='tight' removes the whitespace when saving to file
    fig.savefig(fp, bbox_inches='tight')

if __name__ == '__main__':
    main()
