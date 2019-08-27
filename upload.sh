python setup.py bdist_wheel
twine upload dist/kerasmodels-[*EDIT*].whl

git tag -a v*.*.*
git push origin --tags