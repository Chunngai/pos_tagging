#!/bin/bash

sed 's/Скачать/\./' $1 |
	sed 's/Комментарий/\./' | 
	sed 's/от/ແລ້ວ/' |
	sed 's/⛈/./' |
	sed 's/☀️/./' | 
	sed 's/Видео/ນ້ໍາ/' |
	sed 's/𗂟𗂳𗀄𗀄/\./' |
	sed 's/𗀱𗀱คาวตายเลยเอาไปลาบ𗀄𗀄𗀄𗀗𗀗𗀗/\./' |
	sed 's/\�/\./' |
	sed 's/‍‍‍	/ຖົງ\t/' >  "$1.cleaned"

