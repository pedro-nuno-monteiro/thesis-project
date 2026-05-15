import re
from pathlib import Path

# {"user_01" : {"a00": {"esp_1": path, "esp_2": path, ...},
#            "a01": {...},
#            ... }
#  "user_02" : {...},
#  ...}
FileMap = dict[str, dict[str, dict[str, Path]]]

# user_id-activity_id-place_id-esp_id-timestamp.csv
PATTERN = re.compile(
    r"^(?P<user>\d+)_(?P<activity>\d+)_(?P<place>\d+)_(?P<esp>\d+)_(?P<date>\d{4}-\d{2}-\d{2})\.csv$",
)

PATTERN_OLD = re.compile(r"^(\d+)-([a-zA-Z])(\d+)-(\d+)-(\d+)$")

def sort_meta_info(path: str) -> tuple[list[int], list[str], list[int], list[int]]:

	base: Path = Path(path)
	users_id = set()
	activities_id = set()
	places_id = set()
	esps_id = set()

	for file in base.glob("*.csv"):
		match = PATTERN.match(file.name)
		if not match:
			print("skipped on no match, ", file.name)
			continue

		user = match.group("user")
		activity = match.group("activity")
		place = match.group("place")
		esp = match.group("esp")

		users_id.add(user)
		activities_id.add(activity)
		places_id.add(place)
		esps_id.add(esp)

	return (
		sorted(users_id),
		sorted(activities_id),
		sorted(places_id),
		sorted(esps_id),
	)


# ------------------
# já não é utilizada
# ------------------
def sort_meta_info_old(path: str) -> tuple[list[int], list[str], list[int], list[int]]:

	base: Path = Path(path)
	users_id = set()
	posicoes = set()
	esp_ids = set()
	repetition_ids = set()

	for file in base.glob("*.csv"):
		match = PATTERN_OLD.match(file.stem)
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
# # para cada atividade
# # para cada cenário
# # para cada esp
def get_csv_files_generalistic(path: str) -> FileMap:

	files: FileMap = {}
	base: Path = Path(path)

	# legenda

	# USER ID
	# # 00 - ninguém
	# # 01 - Pepas

	# ACTIVITY
	# # 00 - empty room
	# # 01 - walking

	# # PLACE
	# # 01 - Gab. Pepas
	# # 02 - Gab. Rafa
	# # 03 - Gab. Lab
	# # 04 - Lab.

	for file in base.glob("*.csv"):

		match = PATTERN.match(file.name)
		if not match:
			continue

		user = match.group("user")
		activity = match.group("activity")
		place = match.group("place")
		esp = match.group("esp")

		user_key = f"user_{user}"
		activity_key = f"activity_{activity}"
		place_key = f"place_{place}"
		esp_key = f"esp_{esp}"

		files.setdefault(user_key, {})
		files[user_key].setdefault(activity_key, {})
		files[user_key][activity_key].setdefault(place_key, {})
		files[user_key][activity_key][place_key].setdefault(esp_key, [])

		# append file (support multiple timestamps)
		files[user_key][activity_key][place_key][esp_key].append(file)

	return files


# ------------------
# já não é utilizada
# ------------------
# nesta função importamos todos os ficheiros CSV seguindo,
# # para user/pessoa que testou
# # para cada cenário
# # para cada esp
def get_csv_files_generalistic_old(path: str) -> FileMap:

	files: FileMap = {}
	base: Path = Path(path)

	users_id, posicoes, esp_ids, repetition_ids = sort_meta_info_old(path)

	# legenda - hard coded
	# # users_id
	# # # 0 - dados extra
	# # # 1 - loureiro
	# # # 2 - diana

	# # posições
	# # # z00 - sala vazia
	# # # w01 - sala com pessoa (geral)

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
						files[user_key][esp_key][posicao] = file_path

	return files


# --------------------------------------------------
# já não é utilizada
# apenas no projeto antigo (Wi-Fi-Sensing-[meu])

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