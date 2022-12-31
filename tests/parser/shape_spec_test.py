from eincheck.parser.grammar import create_shape_spec


def test_is_checkable() -> None:
    assert create_shape_spec("i j k").is_checkable({})
    assert create_shape_spec("i *j k").is_checkable({})
    assert create_shape_spec("i j k*").is_checkable({})
    assert not create_shape_spec("i *j k*").is_checkable({})
    assert not create_shape_spec("i *j k*").is_checkable(dict(i=2))
    assert not create_shape_spec("i *j k*").is_checkable(dict(i=4, k=2))
    assert create_shape_spec("i *j k*").is_checkable(dict(j=(2, 3)))
