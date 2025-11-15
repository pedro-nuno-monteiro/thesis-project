from pathlib import Path

FileMap = dict[str, dict[str, str]]


# algumas notas a reter
# # utilizamos keys para evitar repetições de código
# # utilizamos os.path.join em vez de "\\ + " para criar paths
# # utilizamos setdefault para criar dicionários e evitar KeyErrors
def get_csv_files(path: str) -> tuple[FileMap, FileMap, FileMap]:
    files_loureiro: FileMap = {}
    files_afinar: FileMap = {}
    files_diana: FileMap = {}

    # ficheiros sem ninguém
    # # tem 2 instâncias

    # ficheiros com pessoas
    # # apenas 1 instância por posição

    # keys

    afinar_repetition_keys = {
        1: "sem_1",
        2: "sem_2",
        3: "com_1",
        4: "com_2",
    }

    sala_vazia_keys = {
        1: "vazio_1",
        2: "vazio_2",
    }

    user_files_map = {0: files_afinar, 1: files_loureiro, 2: files_diana}

    users: list[int] = [0, 1, 2]

    posicoes: list[str] = [
        "z00", "a00", "a01", "a02",
        "a09", "a10", "a11", "b00",
        "b01", "b02", "b03", "b04",
        "b05", "b06", "b07", "b08",
        "b09", "b10", "b11", "c01",
        "c02", "c03", "c04", "c05",
        "c06", "c07", "c08", "c09",
        "c10", "c11", "d01", "d02",
        "d03", "d04", "d05", "d06",
        "d07", "d08", "d09", "d10",
        "d11", "e04", "e05", "e06",
    ]

    posicoes_afinar: list[str] = ["z00"]

    # convert base path to Path for / operator
    base = Path(path)

    # get files
    for user_id in users:
        user_files = user_files_map[user_id]

        for posicao in posicoes:
            for esp_id in range(1, 5):
                # incluir até à 4ª por causa do ficheiro afinar
                for rep in range(1, 5):
                    # utilizar Path / operator em vez de os.path.join
                    if posicao in posicoes_afinar:
                        file_path = base / f"{user_id:02d}-{posicao}-{esp_id:02d}-{rep:02d}.csv"
                    else:
                        file_path = base / f"{user_id:02d}-{posicao}-{esp_id:02d}-01.csv"

                    if user_id == 0:
                        if posicao not in posicoes_afinar:
                            continue
                        key = afinar_repetition_keys.get(rep)
                        if key:
                            user_files.setdefault(key, {})[f"esp_{esp_id}"] = file_path

                    # loureiro e diana
                    elif posicao == "z00":
                        # posição vazia
                        key = sala_vazia_keys.get(rep)
                        if key:
                            user_files.setdefault(key, {})[f"esp_{esp_id}"] = file_path

    return files_loureiro, files_diana, files_afinar


# Data loading for generic data (without specific the collection for size,
# as it should be the same for all users!!)

def get_csv_files_generalistic(path: str) -> FileMap:

    loaded_files: FileMap = {}

    users = [0, 1, 2]

    posicoes = ['z00', 'a00', 'a01', 'a02', 'a09', 'a10', 'a11', 'b00', 'b01', 'b02',
        'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b09', 'b10', 'b11', 'c01', 'c02',
        'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10', 'c11', 'd01', 'd02',
        'd03', 'd04', 'd05', 'd06', 'd07', 'd08', 'd09', 'd10', 'd11', 'e04', 'e05', 'e06']

    esp_ids = [1, 2, 3, 4]
    repetition_ids = [1, 2, 3, 4, 5]

    base = Path(path)

    # get files
    for user_id in users:
        for pos in posicoes:
            for esp_id in esp_ids:
                for rep in repetition_ids:

                    file_path = base / f"{user_id:02d}-{pos}-{esp_id:02d}-{rep:02d}.csv"

                    if file_path.exists():
                        # key = f"user_{user_id}_{pos}_rep_{rep}"
                        loaded_files.setdefault(pos, {})[f"esp_{esp_id}"] = file_path

    return loaded_files
