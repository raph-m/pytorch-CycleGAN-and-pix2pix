def create_env_file(batch_size="2"):
    content = """
    cuhk = {
        "batch_size": "%s"
    }
    
    flickr = {
        "batch_size:": "%s"
    }
    
    local_params = {
        "cuhk": cuhk,
        "flickr": flickr
    }
    """ % (batch_size, batch_size)

    print(content)

    f = open("env.py", "w")

    f.write(content)
    f.close()


if __name__ == "__main__":
    create_env_file()