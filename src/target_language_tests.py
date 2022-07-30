
import constant

def determine_target_language(trn_langs, val_langs, target_language):
    if target_language:
        assert target_language in val_langs, f"Target language {target_language} was not in validation languages: {val_langs}"
        return target_language
    if not target_language:
        # If it's a pure optimization, then use that language
        if len(trn_langs) == 1:
            return trn_langs[0]
        if len(val_langs) == 1:
            return val_langs[0]
        one_other_target_language = (len(val_langs) == 2 and constant.SUPERVISED_LANGUAGE in val_langs)
        if one_other_target_language:
            # If two languages, favor the non-English language
            not_supervised_languages = list(filter(lambda lang : lang != constant.SUPERVISED_LANGUAGE, val_langs))
            assert len(not_supervised_languages) == 1
            return not_supervised_languages[0]
        # Otherwise, optimize over all unsupervised languages.
        return 'all_unsupervised'

if __name__ == '__main__':
    tests = [
        ('English', determine_target_language(['English'], ['Dutch'], '')),
        ('Dutch', determine_target_language(['English'], ['Dutch'], 'Dutch')),
        ('English', determine_target_language(['English', 'Dutch'], ['English'], '')),
        ('Dutch', determine_target_language(['English', 'Dutch'], ['English', 'Dutch'], '')),
        ('all_unsupervised', determine_target_language(['English', 'Dutch'], ['English', 'Dutch', 'French'], '')),
    ]
    for index, (expected, actual) in enumerate(tests):
        if expected != actual:
            print(index)
            print(expected)
            print(actual)
    