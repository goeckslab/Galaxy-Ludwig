

const FileList = (props) => {

  const {data} = props;
  if (!data) {
    return (
      <>
        <hr />
        <p>No data is found!</p>
        </>
    )
  }

  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const [content, setContent] = React.useState();

  const loadJson = () => {
    fetch(
      data[selectedIndex]["file"],
      {
        headers : { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
         }
      }
    )
    .then( (response) => {
      if (response.ok) {
        return response.json();
      }
      throw new Error("Content not found!");
    })
    .then(json => setContent(json))
    .catch((err) => {
      setContent(null);
      console.log(err);
    })
  };

  React.useEffect( () => {
      if (data[selectedIndex]["type"] === "json") {
        loadJson();
      }
    }, [selectedIndex]
  );

  return (
    <>
      <div className="sidebar">
        <ul>
          {data.map( (item, index) => (
            <li
              key={index}
              className={selectedIndex===index ? "selected": null}
              onClick={() => setSelectedIndex(index)}
            >
              {item.label}
            </li>
          ))}

        </ul>
      </div>
      <div className="selected-container">
        {data[selectedIndex]["type"] === 'image' ? (
          <img src={data[selectedIndex]["file"]} />
        ) : (
          <pre>{JSON.stringify(content, null, 2)}</pre>
        )}
      </div>
    </>
  );
}

const App = (props) => {
  const {title, data} = props;
  return (
    <>
      <h2>{title}</h2>
      <FileList data={data}/>
    </>
  );
}
