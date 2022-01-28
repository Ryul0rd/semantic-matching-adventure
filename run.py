from world import World

def main():
    world = World()
    while world.playing:
        player_action = input('> ')
        if player_action == 'exit':
            world.playing = False
        else:
            world.act(player_action)

if __name__ == '__main__':
    main()