import re
from pathlib import Path

# {"user_01" : {"a00": {"esp_1": path, "esp_2": path, ...},
#            "a01": {...},
#            ... }
#  "user_02" : {...},
#  ...}
FileMap = dict[str, dict[str, dict[str, Path]]]

# user_id-location-esp_id-repetition_id.csv
PATTERN = re.compile(r"^(\d+)-([a-zA-Z])(\d+)-(\d+)-(\d+)$")


def sort_meta_info(path: str) -> tuple[list[int], list[str], list[int], list[int]]:

	base: Path = Path(path)
	users_id = set()
	posicoes = set()
	esp_ids = set()
	repetition_ids = set()

	for file in base.glob("*.csv"):
		match = PATTERN.match(file.stem)
		if not match:
			print("skipped on no match, ", file.name)
			continue

		user, pos_letter, pos_no, esp, rep = match.groups()

		users_id.add(int(user))
		posicoes.add(f"{pos_letter}{int(pos_no):02d}")
		esp_ids.add(int(esp))
		repetition_ids.add(int(rep))

	users_id = sorted(users_id)
	posicoes = sorted(posicoes)
	esp_ids = sorted(esp_ids)
	repetition_ids = sorted(repetition_ids)

	return users_id, posicoes, esp_ids, repetition_ids


# nesta função importamos todos os ficheiros CSV seguindo,
# # para user/pessoa que testou
# # para cada posição
# # para cada esp
def get_csv_files_generalistic(path: str) -> FileMap:

	files: FileMap = {}
	base: Path = Path(path)

    # legenda
    # # 0 - dados extra
    # # 1 - loureiro
    # # 2 - diana
    users_id = [0, 1, 2]

    # z00 - sala vazia
    # w01 - sala com pessoa (geral)
    posicoes = ["z00", "w00", "a00", "a01", "a02", "a09", "a10",
        "a11", "b00", "b01", "b02", "b03", "b04", "b05", "b06",
        "b07", "b08", "b09", "b10", "b11", "c01", "c02", "c03",
        "c04", "c05", "c06", "c07", "c08", "c09", "c10", "c11",
        "d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08",
        "d09", "d10", "d11", "e04", "e05", "e06",
    ]

    esp_ids = [1, 2, 3, 4]
    repetition_ids = [1, 2]

	for user_id in users_id:
		user_key = f"user_{user_id}"
		files[user_key] = {}

		for esp_id in esp_ids:
			esp_key = f"esp_{esp_id}"
			files[user_key][esp_key] = {}

			for posicao in posicoes:
				files[user_key][esp_key][posicao] = None

				for rep in repetition_ids:
					filename = f"{user_id:02d}-{posicao}-{esp_id:02d}-{rep:02d}.csv"
					file_path = base / filename

                    if file_path.exists():
                        files[user_key][posicao][esp_key].append(file_path)

	return files


# --------------------------------------------------
# já não é utilizada

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
		"z00",
		"a00",
		"a01",
		"a02",
		"a09",
		"a10",
		"a11",
		"b00",
		"b01",
		"b02",
		"b03",
		"b04",
		"b05",
		"b06",
		"b07",
		"b08",
		"b09",
		"b10",
		"b11",
		"c01",
		"c02",
		"c03",
		"c04",
		"c05",
		"c06",
		"c07",
		"c08",
		"c09",
		"c10",
		"c11",
		"d01",
		"d02",
		"d03",
		"d04",
		"d05",
		"d06",
		"d07",
		"d08",
		"d09",
		"d10",
		"d11",
		"e04",
		"e05",
		"e06",
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
					else:
						# posição vazia
						if posicao == "z00":
							key = sala_vazia_keys.get(rep)
							if key:
								user_files.setdefault(key, {})[f"esp_{esp_id}"] = (
									file_path
								)
						# posição normal
						else:
							user_files.setdefault(posicao, {})[f"esp_{esp_id}"] = file_path
	return files_loureiro, files_diana, files_afinar

