

const FileList = (props) => {

  const {files} = props;

  return (
    <>
      <ul>
        {files.map( (fl) => (
          <li><a href={fl}>{fl}</a></li>
        ))}

      </ul>
    </>
  );
}
