""" 
Example use case of a doubly linked list representing a playlist of songs. 
Each node in the list contains the name of the song and the artist. 
Provides functionalites to add songs to the playlist, display the playlist, and remove a song by its name.
"""

class Song:
    def __init__(self, title, artist):
        self.title = title
        self.artist = artist

class Node:
    def __init__(self, song):
        self.song = song
        self.prev = None
        self.next = None

class Playlist:
    def __init__(self):
        self.head = None

    def add_song(self, title, artist):
        new_song = Song(title, artist)
        new_node = Node(new_song)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        new_node.prev = current

    def remove_song(self, title):
        current = self.head
        while current:
            if current.song.title == title:
                if current.prev:
                    current.prev.next = current.next
                else:
                    self.head = current.next
                if current.next:
                    current.next.prev = current.prev
                return
            current = current.next
        print(f"Song '{title}' not found in the playlist.")

    def display_playlist(self):
        if self.head is None:
            print("Playlist is empty.")
            return
        current = self.head
        while current:
            print(f"{current.song.title} - {current.song.artist}")
            current = current.next

# Creating a playlist
playlist = Playlist()

# Adding songs to the playlist
playlist.add_song("Song 1", "Artist 1")
playlist.add_song("Song 2", "Artist 2")
playlist.add_song("Song 3", "Artist 3")

# Displaying the playlist
print("Playlist:")
playlist.display_playlist()

# Removing a song from the playlist
playlist.remove_song("Song 2")

# Displaying the playlist after removal
print("\nPlaylist after removing 'Song 2':")
playlist.display_playlist()

"""
    Explanation:
    - We define a `Song` class to represent a song with attributes `title` and `artist`.
    - We have a `Node` class representing a node in the doubly linked list. Each node contains a `song` attribute which is an instance of the `Song` class, and `prev` and `next` pointers.
    - The `Playlist` class manages the playlist using a doubly linked list. It has methods to add a song, remove a song by its title, and display the playlist.
    - In the example, we create a playlist object `playlist` and add three songs to it using the `add_song` method.
    - We then display the playlist using the `display_playlist` method.
    - Next, we remove a song titled "Song 2" from the playlist using the `remove_song` method.
    - Finally, we display the playlist again to see the updated version after the removal.
"""
